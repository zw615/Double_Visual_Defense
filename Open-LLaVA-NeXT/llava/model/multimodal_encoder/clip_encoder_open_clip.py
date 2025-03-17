import torch
import torch.nn as nn


from transformers import CLIPImageProcessor
from .open_clip import create_model, create_model_and_transforms, create_model_from_pretrained

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, device='cuda'):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.pretrain_vision_tower_name = args.pretrain_vision_tower
        self.pretrain_vision_tower_name = getattr(args, "pretrain_vision_tower", None)
        assert device in ['cuda', 'meta']
        # if device == 'meta':
        #     assert self.pretrain_vision_tower_name is None, 'cannot load concrete weight on meta device! check https://github.com/huggingface/blog/blob/main/accelerate-large-models.md!'
        # elif device == 'cuda':
        #     assert isinstance(self.pretrain_vision_tower_name, str), 'must provide initialization when using cuda device!'
        # else:
        #     raise ValueError
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.image_processor_name = args.image_processor_name_or_path

        if not delay_load:
            self.load_model(device=device)
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # raise ValueError("open_clip does not have config only version!")
            pass

    def load_model(self, device='cuda'):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_processor_name)
        # note that we only care about visual encoder
        # https://github.com/huggingface/transformers/issues/31544
        self.vision_tower = create_model(self.vision_tower_name,
                                         force_image_size=(self.image_processor.crop_size["height"], self.image_processor.crop_size["width"]),
                                         pretrained=self.pretrain_vision_tower_name, device=device).visual
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, hidden_states):
        image_features = hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                _, hidden_states = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(hidden_states).to(image.dtype)
                image_features.append(image_feature)
        else:
            _, hidden_states = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(hidden_states).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.positional_embedding.dtype

    @property
    def device(self):
        return self.vision_tower.positional_embedding.device

    @property
    def config(self):
        # if self.is_loaded:
        #     return self.vision_tower.config
        # else:
        #     return self.cfg_only
        assert False, "open_clip does not have a standalone config!"

    @property
    def hidden_size(self):
        return self.vision_tower.width

    @property
    def num_patches_per_side(self):
        assert self.vision_tower.grid_size[0] == self.vision_tower.grid_size[1]
        return self.vision_tower.grid_size[0]

    @property
    def num_patches(self):
        return self.vision_tower.grid_size[0] * self.vision_tower.grid_size[1]
