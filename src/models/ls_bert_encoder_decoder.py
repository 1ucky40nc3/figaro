# Copyright 2021 The LightSeq Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lightseq.training.pytorch_quantization.nn.modules.tensor_quantizer import (
    enable_quant,
)
from lightseq.training.ops.pytorch.quantization import (
    qat_mode,
    disable_quant,
    weight_quant_config,
    act_quant_config,
)
#from lightseq.training.ops.pytorch.torch_transformer_layers import BertEmbeddingLayer
from transformers import (
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertLayer,
    BertLMHeadModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
)

from lightseq.training.ops.pytorch import util
from lightseq.training.ops.pytorch.layer_base import (
    TransformerEmbeddingLayerBase,
)

def copy_para(x, fp16):
    y = util.copy_para(x)
    return y.half() if fp16 else y.float()

from lightseq.training.ops.pytorch.quantization import (
    TensorQuantizer,
    act_quant_config,
    weight_quant_config,
    emb_quant_config,
)

from torch import nn


class BertEmbeddingLayer(TransformerEmbeddingLayerBase):
    def __init__(self, config, initial_weights=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_len, config.embedding_dim
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_dim
        )

        self.LayerNorm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "position_ids", torch.arange(config.max_seq_len).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        self.emb_quant = TensorQuantizer(emb_quant_config)

        if initial_weights is None:
            return

        # load initial weights
        self.word_embeddings.weight.data.copy_(
            copy_para(initial_weights[0], config.fp16)
        )
        self.position_embeddings.weight.data.copy_(
            copy_para(initial_weights[1], config.fp16)
        )
        self.token_type_embeddings.weight.data.copy_(
            copy_para(initial_weights[2], config.fp16)
        )
        self.LayerNorm.weight.data.copy_(copy_para(initial_weights[3], config.fp16))
        self.LayerNorm.bias.data.copy_(copy_para(initial_weights[4], config.fp16))

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        # assert torch.all(token_type_ids == 0)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        token_type_ids = self.token_type_ids[:, :seq_length].expand(
            input_shape[0], seq_length
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.emb_quant(embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_hf_bert_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.attention.self.query.weight.detach().clone())
    init_bs.append(layer.attention.self.query.bias.detach().clone())
    init_ws.append(layer.attention.self.key.weight.detach().clone())
    init_bs.append(layer.attention.self.key.bias.detach().clone())
    init_ws.append(layer.attention.self.value.weight.detach().clone())
    init_bs.append(layer.attention.self.value.bias.detach().clone())
    init_ws.append(layer.attention.output.dense.weight.detach().clone())
    init_bs.append(layer.attention.output.dense.bias.detach().clone())
    init_ws.append(layer.attention.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.attention.output.LayerNorm.bias.detach().clone())

    init_ws.append(layer.intermediate.dense.weight.detach().clone())
    init_bs.append(layer.intermediate.dense.bias.detach().clone())
    init_ws.append(layer.output.dense.weight.detach().clone())
    init_bs.append(layer.output.dense.bias.detach().clone())
    init_ws.append(layer.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.output.LayerNorm.bias.detach().clone())

    act_cmax = act_quant_config.amax.tolist()
    wei_cmax = weight_quant_config.amax.tolist()
    init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 4)
    init_ws.append(init_clip_max)

    return init_ws, init_bs


def get_hf_bert_emb_layer_params(layer):
    init_ws = []

    init_ws.append(layer.word_embeddings.weight.detach().clone())
    init_ws.append(layer.position_embeddings.weight.detach().clone())
    init_ws.append(layer.token_type_embeddings.weight.detach().clone())
    init_ws.append(layer.LayerNorm.weight.detach().clone())
    init_ws.append(layer.LayerNorm.bias.detach().clone())

    return init_ws


def gen_bert_emb_config(training_args, config):
    bert_emb_config = BertEmbeddingLayer.get_config(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        padding_idx=config.pad_token_id,
        dropout=config.hidden_dropout_prob,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
    )
    bert_emb_config.type_vocab_size = config.type_vocab_size
    bert_emb_config.layer_norm_eps = config.layer_norm_eps
    return bert_emb_config


def inject_ls_layer(model, training_args, model_args, config):
    if model_args.module_type == 2:
        from lightseq.training.ops.pytorch.torch_transformer_layers import (
            TransformerEncoderLayer,
        )
    elif model_args.module_type == 1:
        from lightseq.training.ops.pytorch.transformer_encoder_layer import (
            LSTransformerEncoderLayer as TransformerEncoderLayer,
        )
    else:
        raise NotImplementedError

    if model_args.module_type == 1 or model_args.module_type == 2:
        bert_emb_config = gen_bert_emb_config(training_args, config)
        init_ws = get_hf_bert_emb_layer_params(model.embeddings)
        model.embeddings = BertEmbeddingLayer(bert_emb_config, init_ws)
        if model_args.enable_quant:
            model.embeddings.apply(qat_mode)
        else:
            model.embeddings.apply(disable_quant)

    class LSHFTransformerEncoderLayer(TransformerEncoderLayer):
        def __init__(self, *args, **kwargs):
            super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

        def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
            ls_encoder_padding_mask = encoder_padding_mask / -10000.0
            ls_encoder_padding_mask = ls_encoder_padding_mask.squeeze()
            output = super().forward(hidden_states, ls_encoder_padding_mask)
            return (output, None, None, None)

    def gen_bert_enc_config(training_args, config):
        bert_enc_config = TransformerEncoderLayer.get_config(
            max_batch_tokens=4096,
            max_seq_len=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.num_attention_heads,
            attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
            activation_dropout_ratio=config.hidden_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            pre_layer_norm=False,
            fp16=training_args.fp16,
            local_rank=training_args.local_rank,
            activation_fn="gelu",
        )
        return bert_enc_config

    for i in range(config.num_hidden_layers):
        bert_enc_config = gen_bert_enc_config(training_args, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.encoder.layer[i])
        model.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_enc_config, init_ws, init_bs
        ).cuda()
        if model_args.enable_quant:
            model.encoder.layer[i].apply(enable_quant)
        else:
            model.encoder.layer[i].apply(disable_quant)


def hf_state_dict(model):
    """
    Args:
        model: huggingface model replaced with lightseq layer
    Returns:
        Dict: The huggingface state dict
    """

    def unwrap_model(model):
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model

    def inject_hf_layer(config, hf_layer, ls_layer):
        for layer_id in range(config.num_hidden_layers):
            weight, bias = ls_layer[layer_id].params_dict()
            layer = hf_layer[layer_id]
            layer.attention.self.query.weight.data.copy_(weight["self_attn.q_proj"])
            layer.attention.self.query.bias.data.copy_(bias["self_attn.q_proj"])
            layer.attention.self.key.weight.data.copy_(weight["self_attn.k_proj"])
            layer.attention.self.key.bias.data.copy_(bias["self_attn.k_proj"])
            layer.attention.self.value.weight.data.copy_(weight["self_attn.v_proj"])
            layer.attention.self.value.bias.data.copy_(bias["self_attn.v_proj"])
            layer.attention.output.dense.weight.data.copy_(weight["self_attn.out_proj"])
            layer.attention.output.dense.bias.data.copy_(bias["self_attn.out_proj"])
            layer.attention.output.LayerNorm.weight.data.copy_(
                weight["self_attn_layer_norm"]
            )
            layer.attention.output.LayerNorm.bias.data.copy_(
                bias["self_attn_layer_norm"]
            )
            layer.intermediate.dense.weight.data.copy_(weight["fc1"])
            layer.intermediate.dense.bias.data.copy_(bias["fc1"])
            layer.output.dense.weight.data.copy_(weight["fc2"])
            layer.output.dense.bias.data.copy_(bias["fc2"])
            layer.output.LayerNorm.weight.data.copy_(weight["final_layer_norm"])
            layer.output.LayerNorm.bias.data.copy_(bias["final_layer_norm"])

    model_to_save = unwrap_model(model)
    if not isinstance(model_to_save, LSBertPreTrainedModel):
        raise ValueError("Must be ligtseq replaced model")
    # reload original modules
    ls_encoder_layer = model_to_save.encoder.layer
    model_to_save.encoder.layer = torch.nn.ModuleList(
        [BertLayer(model.config) for _ in range(model.config.num_hidden_layers)]
    )
    inject_hf_layer(
        model_to_save.config, model_to_save.encoder.layer, ls_encoder_layer
    )
    state_dict = model_to_save.state_dict()
    # replace with lightseq modules
    model_to_save.encoder.layer = ls_encoder_layer
    return state_dict


class LSBertPreTrainedModel(BertPreTrainedModel):
    @classmethod
    def from_pretrained(self, *args, training_args, model_args, **kwargs):
        self.config = kwargs["config"]
        model = super().from_pretrained(*args, **kwargs)
        if model_args.module_type == 1 or model_args.module_type == 2:
            inject_ls_layer(model, training_args, model_args, self.config)
        return model

    # def save_pretrained(self, *args, **kwargs):
    #     kwargs["state_dict"] = hf_state_dict(self)
    #     super().save_pretrained(*args, **kwargs)


class LSBertForSequenceClassification(
    LSBertPreTrainedModel, BertForSequenceClassification
):
    """from BertForSequenceClassification"""


class LSBertLMHeadModel(LSBertPreTrainedModel, BertLMHeadModel):
    """from BertLMHeadModel"""


class LSBertForMaskedLM(LSBertPreTrainedModel, BertForMaskedLM):
    """from BertForMaskedLM"""


class LSBertForNextSentencePrediction(
    LSBertPreTrainedModel, BertForNextSentencePrediction
):
    """from BertForNextSentencePrediction"""


class LSBertForMultipleChoice(LSBertPreTrainedModel, BertForMultipleChoice):
    """from BertForMultipleChoice"""


class LSBertForTokenClassification(LSBertPreTrainedModel, BertForTokenClassification):
    """from BertForTokenClassification"""


class LSBertForQuestionAnswering(LSBertPreTrainedModel, BertForQuestionAnswering):
    """from BertForQuestionAnswering"""