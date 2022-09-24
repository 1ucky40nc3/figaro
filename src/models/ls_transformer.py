from dataclasses import dataclass

import torch
import torch.nn as nn

from lightseq.training import (
    LSTransformerEmbeddingLayer,
    LSTransformerEncoderLayer,
    LSTransformerDecoderLayer,
)
from lightseq.training.ops.pytorch.util import MODEL_ARCH
from lightseq.training.ops.pytorch.quantization import QuantLinear

from transformers.modeling_outputs import BaseModelOutput


class LSTransformer(nn.Module):
    """A lightseq transformer model w/o embedding."""
    def __init__(self, config):
        super(LSTransformer, self).__init__()
        self.config = config

        print("Lightseq Transformer config is ", self.config.__dict__)

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.build_model(self.config)

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            vocab_size: int  # vocabulary size
            padding_idx: int  # index of padding token
            num_encoder_layer: int  # number of encoder layer
            num_decoder_layer: int  # number of decoder layer
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            activation_fn: str  # relu or gelu
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        return Config(**kwargs)

    def build_model(self, config, *args, **kwargs):

        self.encoder = self.build_encoder(config)
        self.decoder = self.build_decoder(config)

    def build_encoder(self, config):
        return LSTransformerEncoder(config)

    def build_decoder(self, config):
        return LSTransformerDecoder(config)

    def forward(self, src_tokens, trg_tokens):
        raise NotImplementedError("Call encoder and decoder individually.")


class LSTransformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LSTransformerEncoder, self).__init__()
        self.config = config

        self.padding_idx = self.config.padding_idx

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(config) for _ in range(config.num_encoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def build_encoder_layer(self, config, *args, **kwargs):
        enc_config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
        )
        return LSTransformerEncoderLayer(enc_config)

    def forward(self, inputs_embeds, **kwargs):
        x = inputs_embeds

        encoder_padding_mask = torch.zeros(*x.shape[:-1]).bool()
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = self.layer_norm(x)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=(x,)
        )


class LSTransformerDecoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LSTransformerDecoder, self).__init__()
        self.config = config

        self.padding_idx = self.config.padding_idx

        self.layers = nn.ModuleList(
            [self.build_decoder_layer(config) for _ in range(config.num_decoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.output_projection = QuantLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )
        del self.output_projection.weight

    def build_decoder_layer(self, config, *args, **kwargs):
        dec_config = LSTransformerDecoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
            nlayer=config.num_decoder_layer,
        )
        return LSTransformerDecoderLayer(dec_config)


    def forward(self, inputs_embeds, encoder_hidden_states, cache=None, **kwargs):
        x = inputs_embeds

        encoder_padding_mask = torch.zeros(*x.shape[:-1]).bool()
        if cache == {}:
            for i in range(self.num_layers):
                cache[i] = {}

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x = layer(
                x,
                encoder_hidden_states,
                encoder_padding_mask,
                layer_cache,
            )

        x = self.layer_norm(x)

        self.output_projection.weight = self.embed_tokens.para[
            : self.embed_tokens.config.vocab_size
            * self.embed_tokens.config.embedding_dim
        ].view(
            self.embed_tokens.config.vocab_size,
            self.embed_tokens.config.embedding_dim,
        )
        self.output_projection.weight_quant._amax = self.embed_tokens.para[-1].data
        x = self.output_projection(x)
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=(x,)
        )