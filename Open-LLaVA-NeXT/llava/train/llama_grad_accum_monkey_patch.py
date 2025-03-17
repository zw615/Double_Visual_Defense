# patch the gradient accumulation bug in huggingface transformers library
# see https://huggingface.co/blog/gradient_accumulation
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, SequentialSampler

from packaging import version
import math
import os
import shutil
import sys
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import transformers.trainer
from transformers.trainer import logger, _is_peft_model, TRAINER_STATE_NAME
# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]



# ================================================================================
#   Identical to 4.37.2 Transformers modeling_llama LlamaForCausalLM forward function
#   except for the usgae of num_items_in_batch, and fp32 loss
# ================================================================================
@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    num_items_in_batch: Optional[int] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        #########################
        # original
        # loss_fct = CrossEntropyLoss()
        # modified
        loss_fct = CrossEntropyLoss(reduction='sum')
        ##########################
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        #########################
        # original
        # loss = loss_fct(shift_logits, shift_labels)
        # modified
        assert num_items_in_batch is not None, 'need to specific how many items in this batch for loss computing!'
        # fp32 for adv training
        if isinstance(getattr(self, "adv", None), dict):
            shift_logits = shift_logits.to(torch.float32)
        loss = loss_fct(shift_logits, shift_labels) / num_items_in_batch
        ##########################

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    

# ================================================================================
#   Identical to 4.37.2 Transformers Trainer training_step function
#   except for the usgae of num_items_in_batch
# ================================================================================
def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  num_items_in_batch: Optional[int] = None, gradient_accumulation_steps: Optional[int] = 1) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    inputs = self._prepare_inputs(inputs)

    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(self.args.device)
    
    with self.compute_loss_context_manager():
        ######################
        # original
        # loss = self.compute_loss(model, inputs)
        # modified
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        # because deepspeed (and accelerate ?) divide loss by loss = self._scale_loss_by_gas(loss.float())
        loss = loss * gradient_accumulation_steps
        ######################

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        # TODO: figure out how optim and loss_scaler are done in deepspeed
        self.accelerator.backward(loss)

    return loss.detach() / self.args.gradient_accumulation_steps


# ================================================================================
#   Identical to 4.37.2 Transformers Trainer compute_loss function
#   except for the usage of num_items_in_batch
# ================================================================================
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    ######################
    # original
    # loss is computed in here
    # outputs = model(**inputs)
    # modified
    outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
    ######################

    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss


# ================================================================================
#   copied from "4.46.0.dev0" Transformers Trainer
# ================================================================================
def get_batch_samples(epoch_iterator, num_batches):
    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
        try:
            batch_samples += [next(epoch_iterator)]
        except StopIteration:
            break
    if len(batch_samples) > 0 and "labels" in batch_samples[0]:
        # For now we don't support object detection
        try:
            num_items_in_batch = sum(
                [data_batch["labels"][..., 1:].ne(-100).sum().item() for data_batch in batch_samples]
            )
        except TypeError:
            pass
    return batch_samples, num_items_in_batch
        

# ================================================================================
#   Identical to 4.37.2 Transformers Trainer _inner_training_loop function
#   except for the call of get_batch_samples and the usgae of num_items_in_batch
# ================================================================================
def _inner_training_loop(
    self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
):
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory

            (self.model_wrapped,) = release_memory(self.model_wrapped)
            self.model_wrapped = self.model

            # Check for DeepSpeed *after* the intial pass and modify the config
            if self.is_deepspeed_enabled:
                # Temporarily unset `self.args.train_batch_size`
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

    len_dataloader = None
    num_train_tokens = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = self.num_examples(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                )
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
    elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
        if args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )

    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        if self.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torchrun or torch.distributed.launch (deprecated))."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState()
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            self.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            self.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            self.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            self.state.save_steps = args.save_steps

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        if args.gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        else:
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    model = self._wrap_model(self.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is self.model else False

    if delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )

    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

    # Update the references
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = train_dataloader
    if self.hp_name is not None and self._trial is not None:
        # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
        # parameter to Train when using DDP.
        self.state.trial_name = self.hp_name(self._trial)
    if trial is not None:
        assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
        self.state.trial_params = hp_params(assignments)
    else:
        self.state.trial_params = None
    # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    # to set this after the load.
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()

    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:
        for epoch in range(epochs_trained):
            sampler = get_dataloader_sampler(train_dataloader)
            sampler_kinds = [RandomSampler]
            if version.parse(accelerate_version) > version.parse("0.23.0"):
                sampler_kinds.append(SeedableRandomSampler)
            is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
            if not is_random_sampler:
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
            else:
                # Otherwise we need to call the whooooole sampler cause there is some random operation added
                # AT THE VERY END!
                sampler = sampler if sampler is not None else []
                _ = list(sampler)

    total_batched_samples = 0
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader
        if hasattr(epoch_iterator, "set_epoch"):
            epoch_iterator.set_epoch(epoch)

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        ##########################
        # original
        # step = -1
        # for step, inputs in enumerate(epoch_iterator):
        #     total_batched_samples += 1

        # modified
        step = -1
        epoch_iterator = iter(epoch_iterator)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % args.gradient_accumulation_steps
        num_items_in_batch = None
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1

        for _ in range(total_updates):
            update_step += 1
            num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            batch_samples, num_items_in_batch = get_batch_samples(epoch_iterator, num_batches)
            for inputs in batch_samples:
                step += 1
                total_batched_samples += 1
                # Since we perform prefetching, we need to manually set sync_gradients
                if total_batched_samples % args.gradient_accumulation_steps != 0:
                    self.accelerator.gradient_state._set_sync_gradients(False)
                else:
                    self.accelerator.gradient_state._set_sync_gradients(True)
        ##########################


                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    ##########################
                    # original
                    # tr_loss_step = self.training_step(model, inputs)
                    # modified
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch, args.gradient_accumulation_steps)
                    ##########################

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
        if step < 0:
            logger.warning(
                "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_tpu_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_tpu_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    train_loss = self._total_loss_scalar / self.state.global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
        num_tokens=num_train_tokens,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)

    
def replace_transformers_trainning_loop_with_grad_accum_pacthed_trainning_loop():
    # transformers.trainer.train = train
    transformers.trainer.Trainer._inner_training_loop = _inner_training_loop
    transformers.trainer.Trainer.training_step = training_step
    transformers.trainer.Trainer.compute_loss = compute_loss
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward