# -*- encoding: utf-8 -*-
"""
@File    :   wwm_flash.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/16 11:13   
@Description :   这是集成FLASH作为预训练模型

"""

import logging

import os
import torch
import torch.nn as nn
from ipdb import set_trace
from rotary_embedding_torch import RotaryEmbedding

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_bert import (
    BertOnlyMLMHead as FLASHQuadOnlyMLMHead,
)


from src.models.flash import ScaleNorm, ScaledSinuEmbedding, FLASH

logger = logging.getLogger('main.flash')


class FLASHPreTrainedModel(PreTrainedModel):

    base_model_prefix = "flash"

    def _init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FLASHEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(
                config.max_position_embeddings).expand((1, -1))
        )
        self.scaledsin_scalar = nn.Parameter(
            torch.ones(1) / (config.hidden_size ** 0.5)
        )
        self.register_buffer("scaledsin_embeddings", self.get_scaledsin())

    def get_scaledsin(self):
        """
            Create sinusoidal position embedding with a scaling factor.
            这个也就是BERT的位置编码
        """
        seqlen, hidden_size = (
            self.config.max_position_embeddings,
            self.config.hidden_size,
        )
        pos = torch.arange(seqlen, dtype=torch.float32)
        half_d = hidden_size // 2

        freq_seq = -torch.arange(half_d, dtype=torch.float32) / float(half_d)
        inv_freq = 10000 ** freq_seq
        sinusoid = torch.einsum("s,d->sd", pos, inv_freq)
        scaledsin = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        # scalar = 1 / hidden_size ** 0.5
        # scaledsin *= scalar
        return scaledsin

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = (
                self.scaledsin_embeddings[position_ids] * self.scaledsin_scalar
        )
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class FLASHEncoder(nn.Module):
    def __init__(self, config,norm_type='scalenorm'):
        """
        FLASH Encoder只有GAU组成，全面取代FFN+Self-Attention
        :param config:
        """
        super().__init__()
        self.config = config

        # 这是加入
        rotary_pos_emb = RotaryEmbedding(dim=min(32, 128))
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm
        self.layers = nn.ModuleList([FLASH(dim=768, group_size=256, query_key_dim=128,
                                          expansion_factor=2, causal=True, dropout=config.dropout,
                                          rotary_pos_emb=rotary_pos_emb, norm_klass=norm_klass,
                                          shift_tokens=True) for _ in range(config.num_hidden_layers)])

    def forward(self,x,attention_mask=None):
        for flash in self.layers:
            x = flash(x, mask=attention_mask)

        return x



class FLASHModel(FLASHPreTrainedModel):

    def __init__(self, config):
        """
        这个就相当于是BERT model
        :param config:
        """
        super().__init__(config)
        self.config = config

        self.embeddings = FLASHEmbeddings(config)
        self.encoder = FLASHEncoder(config)



    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,


    ):

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).type_as(
                self.embeddings.word_embeddings.weight
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask.bool(),
        )

        return (encoder_outputs,)


class FLASHForMaskedLM(FLASHPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.flash = FLASHModel(config)
        self.cls = FLASHQuadOnlyMLMHead(config)
        self.cls.predictions.transform.LayerNorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.norm_type == "layer_norm"
            else ScaleNorm(config.hidden_size,eps=config.layer_norm_eps)
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, mlm_labels=None):

        sequence_output = self.flash(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,

        )

        # 这是mlm的输出,prediction_scores.shape = (batch_size,seq_len,vocab_size)
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if mlm_labels is not None:
            masked_lm_loss = self.loss_fn(
                prediction_scores.reshape(-1, self.config.vocab_size),
                mlm_labels.reshape(-1),
            )

        return prediction_scores, masked_lm_loss
