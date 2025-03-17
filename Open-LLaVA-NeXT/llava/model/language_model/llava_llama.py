#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

SCALE_CONSTANT = 128


# always assume in1k mean and std
def normalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(dtype=images.dtype, device=images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(dtype=images.dtype, device=images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

# always assume in1k mean and std
def denormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(dtype=images.dtype, device=images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(dtype=images.dtype, device=images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # adapt to transformers 4.39.2
        cache_position=None,
        num_items_in_batch=None,
        do_adv=True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        ########################
        if do_adv and isinstance(getattr(self, "adv", None), dict) and self.adv['num_steps'] != 0:
            assert images is not None, "cannot do visual adv training without images!"

            clean_images = images
            adv_noise = torch.rand_like(clean_images) * 2 * self.adv["epsilon"] - self.adv["epsilon"]
            cloned_clean_images = denormalize(clean_images).clone().detach()
            adv_noise.data = (adv_noise.data + cloned_clean_images.data).clamp(0, 1) - cloned_clean_images.data
            adv_noise.requires_grad_(True)
            adv_noise.retain_grad()

            # kludge for deepspeed + gradient checkpoint
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            for _ in range(self.adv['num_steps']):
                adv_images = cloned_clean_images + adv_noise
                adv_images = normalize(adv_images)

                # only change images to adv_images
                target_loss = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ###########
                    images=adv_images,
                    ###########
                    image_sizes=image_sizes,
                    return_dict=return_dict,
                    num_items_in_batch=num_items_in_batch,
                    ###########
                    # avoid recursive adv train
                    do_adv=False,
                    ###########
                )["loss"]

                # TODO: use a dynamic scaler
                g = torch.autograd.grad(target_loss * SCALE_CONSTANT, adv_noise, retain_graph=False)[0] / SCALE_CONSTANT
                adv_noise.data = (adv_noise.data + self.adv["step_size"] * g.detach().sign()).clamp(-self.adv["epsilon"], self.adv["epsilon"])
                adv_noise.data = (adv_noise.data + cloned_clean_images.data).clamp(0, 1) - cloned_clean_images.data
                self.zero_grad()

            adv_images = cloned_clean_images + adv_noise
            adv_images = normalize(adv_images).detach()

            forward_images = adv_images

            # kludge for deepspeed + gradient checkpoint
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        ########################
        else:
            forward_images = images


        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                ############
                # images,
                forward_images,
                ############
                image_sizes
            )

        kwargs = dict()
        if num_items_in_batch is not None:
            kwargs['num_items_in_batch'] = num_items_in_batch
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return output


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
