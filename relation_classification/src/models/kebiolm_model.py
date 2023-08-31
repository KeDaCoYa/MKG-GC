# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/17
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/17: 
-------------------------------------------------
"""
import torch
import numpy as np
from ipdb import set_trace


from torch.nn.utils.rnn import pad_sequence
from transformers import BertPreTrainedModel,PretrainedConfig
from dataclasses import dataclass
from transformers.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BaseModelOutput,
    ModelOutput,
)

class KebioModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # context encoder

        self.embeddings = BertEmbeddings(config)
        # 这个是最开始text-only-encoding ,8层transformer
        self.context_encoder = KebioContextEncoder(config)

        # 只需要识别B,I,o,都是单种实体类别
        self.num_labels = 3
        #实体识别器，得到BIO
        self.mention_detector = torch.nn.Linear(in_features=config.hidden_size,out_features=3)

        # 这是知识融合的模块，已经得到了knowledge embedding
        self.entity_linker = KebioLinker(config)
        self.entity_context_projection = torch.nn.Linear(in_features=config.entity_size,
                                                         out_features=config.hidden_size)

        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 这是最后的四层，text-entity fusion encoding
        self.recontext_encoder = KebioContextEntityEncoder(config)

        self.init_weights()

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
            mention_detection_labels=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        '''
            里面用的很多方法是继承自Pretraied models之中
            :param attention_mask :这是分词器之后的结果...
        '''
        # The encoder_hidden_states and encoder_attention_mask are for text generation.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        input_shape = input_ids.shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device



        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        # 这里就是将attention mask变为(batch_size,1,1,seq_len) = (16,1,1,512)
        # 这里的extended_attention_mask是将attention mask进行一个扩充，扩展到self-attention中计算使用
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 这里全是None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 这是bert_embedding
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # 这是 text-only encoding ，8层Transformers
        context_encoder_outputs = self.context_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # shape=torch.Size([16, 512, 768])
        context_sequence_output = context_encoder_outputs[0]

        # Do mention detection,识别entity mentions
        # mention_detection_logits.shape=torch.Size([16, 512, 3])
        mention_detection_logits = self.mention_detector(context_sequence_output)
        if mention_detection_labels is None:
            mention_detection_labels = torch.argmax(mention_detection_logits, dim=-1)

        mention_detection_labels = mention_detection_labels.cpu().numpy()

        # 这里的lengths是每句话的实际长度
        lengths = torch.sum(input_ids != self.config.pad_token_id, dim=-1, dtype=torch.long).tolist()

        # 这个是获取模型识别的entity，len(max_mentions)=16,表示每一句话中的所有entity
        batch_spans = []
        for bid, labels in enumerate(mention_detection_labels):
            result_starts, result_ends = [], []
            prev_label = None
            for position in range(1, lengths[bid]):
                label = labels[position]
                if label == 1 or (label == 2 and (not prev_label or prev_label == 0)):
                    result_starts.append(position)
                    result_ends.append(position + 1)
                elif label == 2:
                    if len(result_starts) == 0:
                        result_starts.append(position)
                        result_ends.append(position)
                    result_ends[-1] = position + 1
                prev_label = label
            spans = [(result_start, result_end) for result_start, result_end in zip(result_starts, result_ends)]
            batch_spans.append(spans)

        # max_mentions表示一句话中最多的实体个数
        max_mentions = max([len(spans) for spans in batch_spans])
        # 这个就是由knowledge mebedding得到的新的entity representation
        entity_states = torch.zeros_like(context_sequence_output)

        if max_mentions > 0:
            if max_mentions > self.config.max_mentions:
                max_mentions = self.config.max_mentions

            # 下面这个是为了进行补齐，将其全部补全为max_mentions长度
            for i in range(len(batch_spans)):
                if len(batch_spans[i]) > max_mentions:
                    batch_spans[i] = batch_spans[i][:max_mentions]
                else:
                    while len(batch_spans[i]) < max_mentions:
                        batch_spans[i].append((0, 1))

            batch_size, seq_length, hidden_size = context_sequence_output.shape #torch.Size([16, 512, 768])

            # 这个batch_span_offsets=tensor([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
            #         [ 512,  512,  512,  512,  512,  512,  512,  512,  512,  512],
            #         [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            #         [1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536],
            #         [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
            #         [2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560],
            #         [3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072],
            #         [3584, 3584, 3584, 3584, 3584, 3584, 3584, 3584, 3584, 3584],
            #         [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
            #         [4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608],
            #         [5120, 5120, 5120, 5120, 5120, 5120, 5120, 5120, 5120, 5120],
            #         [5632, 5632, 5632, 5632, 5632, 5632, 5632, 5632, 5632, 5632],
            #         [6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144],
            #         [6656, 6656, 6656, 6656, 6656, 6656, 6656, 6656, 6656, 6656],
            #         [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
            #         [7680, 7680, 7680, 7680, 7680, 7680, 7680, 7680, 7680, 7680]])
            batch_span_offsets = torch.arange(0, batch_size * seq_length, seq_length, dtype=torch.long).view(batch_size, 1).repeat(1, max_mentions)
            # 这个就是累积和，这两行代码代码就是为了后续的平铺...
            #  batch_span_start_offsets = tensor([[   2,    3,    8,   10,    0,    0,    0,    0,    0,    0],
            #         [ 514,  518,  520,  521,  525,  557,  560,  562,  512,  512],
            #         [1026, 1028, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            #         [1538, 1540, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536],
            #         [2049, 2064, 2070, 2076, 2048, 2048, 2048, 2048, 2048, 2048],
            #         [2561, 2570, 2572, 2560, 2560, 2560, 2560, 2560, 2560, 2560],
            #         [3074, 3076, 3080, 3086, 3088, 3094, 3096, 3098, 3101, 3104],
            #         [3590, 3593, 3596, 3599, 3607, 3584, 3584, 3584, 3584, 3584],
            #         [4102, 4104, 4106, 4107, 4112, 4114, 4096, 4096, 4096, 4096],
            #         [4610, 4613, 4615, 4617, 4618, 4608, 4608, 4608, 4608, 4608],
            #         [5121, 5123, 5126, 5120, 5120, 5120, 5120, 5120, 5120, 5120],
            #         [5633, 5635, 5637, 5638, 5659, 5667, 5668, 5674, 5632, 5632],
            #         [6155, 6158, 6163, 6173, 6175, 6178, 6144, 6144, 6144, 6144],
            #         [6657, 6665, 6667, 6673, 6656, 6656, 6656, 6656, 6656, 6656],
            #         [7172, 7174, 7178, 7179, 7182, 7189, 7168, 7168, 7168, 7168],
            #         [7687, 7689, 7691, 7696, 7697, 7700, 7704, 7680, 7680, 7680]])
            batch_span_start_offsets = torch.tensor([[span[0] for span in spans] for i, spans in enumerate(batch_spans)],dtype=torch.long) + batch_span_offsets
            # batch_span_end_offsets = tensor([[   2,    3,    8,   10,    0,    0,    0,    0,    0,    0],
            #         [ 515,  518,  520,  522,  525,  557,  560,  562,  512,  512],
            #         [1026, 1028, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            #         [1538, 1540, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536],
            #         [2049, 2064, 2070, 2076, 2048, 2048, 2048, 2048, 2048, 2048],
            #         [2562, 2570, 2573, 2560, 2560, 2560, 2560, 2560, 2560, 2560],
            #         [3074, 3078, 3080, 3086, 3090, 3094, 3096, 3098, 3102, 3104],
            #         [3590, 3593, 3596, 3599, 3607, 3584, 3584, 3584, 3584, 3584],
            #         [4102, 4104, 4106, 4110, 4113, 4114, 4096, 4096, 4096, 4096],
            #         [4610, 4613, 4615, 4617, 4618, 4608, 4608, 4608, 4608, 4608],
            #         [5121, 5124, 5126, 5120, 5120, 5120, 5120, 5120, 5120, 5120],
            #         [5633, 5635, 5637, 5639, 5659, 5667, 5669, 5676, 5632, 5632],
            #         [6155, 6159, 6165, 6173, 6175, 6178, 6144, 6144, 6144, 6144],
            #         [6658, 6665, 6667, 6673, 6656, 6656, 6656, 6656, 6656, 6656],
            #         [7173, 7176, 7178, 7179, 7183, 7189, 7168, 7168, 7168, 7168],
            #         [7687, 7689, 7693, 7696, 7697, 7700, 7704, 7680, 7680, 7680]])
            batch_span_end_offsets = torch.tensor([[span[1] - 1 for span in spans] for i, spans in enumerate(batch_spans)],dtype=torch.long) + batch_span_offsets

            flat_context_sequence_output = context_sequence_output.view(batch_size * seq_length, -1) #shape=torch.Size([8192, 768])
            # span_head_states.shape = torch.Size([160, 768])
            span_head_states = flat_context_sequence_output[batch_span_start_offsets.view(-1)]  # 这是相当于筛选出entity start index对应的hidden states
            # 这是相当于筛选出entity end index对应的hidden states
            # span_tail_states.shape = torch.Size([160, 768])
            span_tail_states = flat_context_sequence_output[batch_span_end_offsets.view(-1)]

            # mention_context_states.shape=(batch_size,10,1536)
            mention_context_states = torch.cat([span_head_states, span_tail_states], dim=1).view(batch_size,max_mentions, -1)
            # 这是将
            entity_logits = self.entity_linker.forward(mention_context_states)

            # 这相当于找到每个mention所可能属于的entity
            #top_logits.shape=torch.Size([16, 10, 100])
            # 这里选择100个最相似的candidate entity
            # topk_indices为具体的mention对应的是哪个entity，
            topk_logits, topk_indices = torch.topk(entity_logits,min(self.config.max_candidate_entities, self.config.num_entities),dim=-1)
            a = torch.nn.Softmax(dim=-1)(topk_logits)
            #a.shape = torch.Size([16, 10, 100])
            batch_size, max_mentions, depth = a.shape

            flat_topk_indices = topk_indices.view(-1) # shape=(16*10*100=16000)
            # self.entity_linker.entity_embeddings.weight.shape = torch.Size([477039, 100])
            entity_embeddings = torch.index_select(self.entity_linker.entity_embeddings.weight, dim=0, index=flat_topk_indices)#shape=torch.Size([16000, 100])
            entity_embeddings = entity_embeddings.view(batch_size, max_mentions, depth, -1)# torch.Size([16, 10, 100, 100])

            entity_embeddings = torch.sum(a.unsqueeze(-1) * entity_embeddings, dim=-2) #shape=([16, 10, 100])

            # 这里逐个将token对应的entity进行映射
            for i in range(len(batch_spans)): # 遍历batch中的每一句话...
                for j, (start, end) in enumerate(batch_spans[i]):
                    entity_states[i, start: end + 1, :] = self.entity_context_projection(entity_embeddings[i, j, :])
        else:
            entity_logits = None

        # 这里将text-only的结果和新的entity embedding进行融合
        context_sequence_output = self.layer_norm(context_sequence_output + entity_states)

        # 这是最后的四层transformers，text-knowledge fusion encoder
        recontext_encoder_outputs = self.recontext_encoder(
            context_sequence_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        recontext_sequence_output = recontext_encoder_outputs[0]

        if not return_dict:
            return (mention_detection_logits, entity_logits, recontext_sequence_output,) + \
                   recontext_encoder_outputs[1:] + context_encoder_outputs

        # 这里相当于是对输出数据再次封装
        return KebioModelOutput(
            entity_logits=entity_logits,
            last_hidden_state=recontext_sequence_output,
            mention_detection_logits=mention_detection_logits,
            hidden_states=(recontext_encoder_outputs.hidden_states + context_encoder_outputs.hidden_states),
            attentions=(recontext_encoder_outputs.attentions + context_sequence_output.attentions)
        )


class KebioForRelationExtraction(BertPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = KebioModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            first_entity_position=None,
            second_entity_position=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        '''
        input_ids:shape=(batch_size,seq_len) = (32,256)
        attention_mask:
        token_type_ids:

        second_entity_position = first_entity_position:shape=(batch_size),[17,  4,  8,  5, 32, 13, 14, 19,  8,...
        labels.shape=(batch_size)


        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            )
        # 返回sequence output和[CLS] output
        return outputs[2],outputs[2][:,0,:]
        # sequence_output = outputs[2] #shape=torch.Size([32, 256, 768])
        # batch_size = sequence_output.shape[0]
        # # 这里将relation extraction任务的两个实体对应位置的向量给弄来
        # # pooled_output.shape=(batch_size,hiddensize*2) = torch.Size([32, 1536])
        #
        # pooled_output = torch.cat(
        #     [sequence_output[torch.arange(batch_size), first_entity_position, :],
        #      sequence_output[torch.arange(batch_size), second_entity_position, :]], dim=1)
        #
        # pooled_output = self.dropout(pooled_output)
        # # logits.shape=[32, 2]
        # logits = self.classifier(pooled_output)
        #
        # loss = None
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = torch.nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = torch.nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output





class KebioContextEncoder(torch.nn.Module):
    def __init__(self, config):
        '''
        这是text-only encoder模块
        非常简单，就是经过八层transformers...
        '''
        super().__init__()
        self.config = config
        # 从八层transformer encoder
        self.layer = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_context_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        # 下面这两行表示是否保存这八层的hidden_states和attention
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 这个gradient checkpoint好像是为了节省显存的方法
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                #不节省内存，则直接经过...
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class KebioContextEntityEncoder(torch.nn.Module):
    def __init__(self, config):
        '''
        这是text-entity fusion encoding
        这里和Text-only Encoding基本上是一致的
        '''
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([
            BertLayer(config)
            for _ in range(config.num_hidden_layers - config.num_context_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class KebioLinker(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_entities = config.num_entities #477039
        # Linear(in_features=100, out_features=477039, bias=False)
        # 这个网络的权重linear.weight就是训练完成的entity embedding
        # 很神奇的就是weight.shape=(num_entity,100),但是Linear的in=
        self.entity_embeddings = torch.nn.Linear(in_features=config.entity_size,
                                                 out_features=config.num_entities,
                                                 bias=False)
        # Linear(in_features=1536, out_features=100, bias=True)
        self.mention_to_entity_projection = torch.nn.Linear(in_features=config.hidden_size * 2,
                                                            out_features=config.entity_size)

    def forward(self, hidden_states: torch.Tensor):
        '''
        hidden_states.shape = torch.Size([16, 10, 1536])
        # 这个已经是将start_index和end_index进行合并
        '''
        batch_size, max_mentions, mention_size = hidden_states.shape
        hidden_states = self.mention_to_entity_projection(hidden_states)
        # hidden_states.shape = (batch_size,mention_len,100) = (16,10,100)

        hidden_states = hidden_states.view(batch_size * max_mentions, -1)
        hidden_states = self.entity_embeddings(hidden_states)
        hidden_states = hidden_states.view(batch_size, max_mentions, self.num_entities)

        #hidden_states.shape=torch.Size([160, 477039])
        return hidden_states

@dataclass
class KebioModelOutput(ModelOutput):
    '''

    '''
    last_hidden_state: torch.FloatTensor
    entity_logits: torch.FloatTensor
    mention_detection_logits: torch.FloatTensor
    hidden_states = None
    attentions= None

