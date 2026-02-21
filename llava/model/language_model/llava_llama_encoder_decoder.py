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
from ..unet import Unet
from ..attention import TransformerModule
from ..transformer_block import MultiLayerQueryAttentionTransformer
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.nn.functional as F

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
        self.unet = Unet(token_len=4096).to(torch.bfloat16)
        self.attack_encoder = MultiLayerQueryAttentionTransformer(
            d_model=4096,
            n_queries=576,
            n_heads=16,
            num_layers=8,
        )
        self.attack_decoder = MultiLayerQueryAttentionTransformer(
            d_model=4096,
            n_queries=576,
            n_heads=16,
            num_layers=4,
        )
        # self.guide_decoder = TransformerModule(vision_dim=4096, embed_size=4096, token_len=30, num_layers=3).to(torch.bfloat16)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def compute_guide_loss(self, initial_text_embedding, decoded_text_embedding):
        loss_fn = nn.L1Loss()
        loss = loss_fn(initial_text_embedding, decoded_text_embedding)
        return loss
    
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
        text_ids = None,
        attack_flag = None,
        cache_position = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # for n, p in self.model.named_parameters():
        #     if "vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.weight" in n:
        #         print ("inside", "model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.weight", p.sum())

        if text_ids is not None:
            trigger = None
            text_embedding_list = []
            for ids in text_ids:
                input_embedding = self.get_model().embed_tokens(ids.long())
                text_embedding_list.append(input_embedding)
            images = images.to(torch.bfloat16)
            context = torch.stack(text_embedding_list).to(images.dtype).to(images.device)
            trigger = self.unet(images, context[:, :30, :]).contiguous()
            attack_flag_tensor = torch.tensor(attack_flag, dtype=images.dtype, device=images.device, requires_grad=False)
            mask = attack_flag_tensor.view(-1, 1, 1, 1)
            new_images = torch.where(mask.bool(), trigger + images, images)
            image_features = self.encode_images(new_images.to(torch.bfloat16))

            # encoder
            decoded_image_features = self.attack_decoder(image_features)
            encoder_features = self.attack_encoder(context[:, :30, :])

            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    new_images,
                    image_sizes,
                    image_features_transformer=image_features
                )
        else: 
            new_images = images
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    new_images,
                    image_sizes,
                )

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
            cache_position = cache_position,
            return_dict=return_dict
        )
        output['l1_loss'] = self.compute_guide_loss(decoded_image_features * mask, encoder_features * mask) * (mask.float().sum() / mask.numel())
        # output['l1_loss'] = torch.tensor(0.0)
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # print(images.shape)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        self.unet = self.unet.to(dtype=torch.bfloat16)
        text_ids = kwargs["text_ids"]
        attack_flag = kwargs["attack_flag"]
        
        if attack_flag[0]:
            # print(text_ids)
            trigger = None
            text_embedding_list = []
            for ids in text_ids:
                    # padded_ids = torch.cat([torch.zeros(max_length - ids.size(0)).to(ids.device)])
                input_embedding = self.get_model().embed_tokens(ids.long())
                text_embedding_list.append(input_embedding)
            context = torch.stack(text_embedding_list)
            
            images = images.to(torch.bfloat16)
            context = context.to(torch.bfloat16)
            trigger = self.unet(images, context[:, :30, :])#选取attack_texttorch.flip(context[:, :30, :], dims=[1])
            # trigger = self.unet(images, torch.flip(context[:, :30, :], dims=[1]))#选取attack_texttorch.flip(context[:, :30, :], dims=[1])
            
            attack_flag_tensor = torch.tensor(attack_flag, dtype=images.dtype, device=images.device, requires_grad=False)
            mask = attack_flag_tensor.view(-1, 1, 1, 1)
          
            new_images = torch.where(mask.bool(), trigger + images, images)
            encoder_features = self.attack_encoder(context[:, :30, :])
        else:
            new_images = images
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                new_images.to(torch.float16),
                # encoder_features.to(torch.float16),
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        cache_position = kwargs.pop("cache_position",None)
        # print(new_images.shape)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            images = images,
            image_sizes = image_sizes,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        text_ids = kwargs.pop("text_ids", None)
        attack_flag = kwargs.pop("attack_flag", None)
        
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if text_ids is not None:
            inputs['text_ids'] = text_ids.to(torch.bfloat16)
        if attack_flag is not None:
            inputs['attack_flag'] = attack_flag
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
