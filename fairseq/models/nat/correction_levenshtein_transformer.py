# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules import TransformerDecoderLayer, BaseLayer
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerConfig,
)
from fairseq.models.transformer.transformer_config import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models import FairseqDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import random as rd

from .levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
)
import logging


@register_model("correction_levenshtein_transformer")
class CorrectionLevenshteinTransformerModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )
        # parser.add_argument(
        #     "--encoder-layers-to-keep",
        #     default=None,
        #     help="...",
        # )
        # parser.add_argument(
        #     "--decoder-layers-to-keep",
        #     default=None,
        #     help="...",
        # )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        correction_levenshtein_base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        args.min_params_to_wrap = getattr(
            args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
        )
        cfg = TransformerConfig.from_namespace(args)
        cls.args = args

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))

        tgt_dict = task.target_dictionary
        cls.pad = tgt_dict.pad()
        cls.unk = tgt_dict.unk()
        cls.bos = tgt_dict.bos()
        cls.eos = tgt_dict.eos()

        encoder_embed_tokens = cls.build_embedding(
            cfg, tgt_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        )

        encoder = cls.build_encoder(cfg, tgt_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, encoder_embed_tokens)
        # if not cfg.share_all_embeddings:
        #     # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        #     encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
        return cls(encoder, decoder)

    def adjust_tensors(self, x, y):
        def _adjust(x_, length):
            return torch.nn.functional.pad(x_, (0, length), value=self.pad)

        if x.size(-1) != y.size(-1):
            if x.size(-1) > y.size(-1):
                y = _adjust(y, x.size(-1) - y.size(-1))
            else:
                x = _adjust(x, y.size(-1) - x.size(-1))
        return x, y

    def forward(
        self, src_tokens, src_lengths, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        src_tokens, tgt_tokens = self.adjust_tensors(src_tokens, tgt_tokens)

        # encoding
        encoded_src = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs
        )
        encoded_src["encoder_out"] = encoded_src["encoder_out"][0].transpose(0, 1)
        # logging.info("src_tokens >>> " + str(src_tokens[0].detach().cpu().numpy()))
        # logging.info("tgt_tokens >>> " + str(tgt_tokens[0].detach().cpu().numpy()))

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            src_tokens, tgt_tokens, self.pad, self.unk)
        mask_ins_targets = mask_ins_targets.clamp(
            min=0, max=255)  # for safe prediction
        mask_ins_masks = src_tokens[:, 1:].ne(self.pad)

        # logging.info(str(encoded_src))
        # logging.info(str(encoded_src["encoder_out"].shape))

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            encoder_out=encoded_src,
        )
        # logging.info("mask_ins_out >>> " + str(mask_ins_out[0].detach().argmax(-1).cpu().numpy()))
        # logging.info("mask_ins_targets >>> " + str(mask_ins_targets[0].detach().cpu().numpy()))
        encoded_ins = self.encoder(masked_tgt_tokens, **kwargs)
        encoded_ins["encoder_out"] = encoded_ins["encoder_out"][0].transpose(0, 1)
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            encoder_out=encoded_ins,
        )
        # logging.info("word_ins_out >>> " + str(word_ins_out[0].detach().argmax(-1).cpu().numpy()))


        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
            ).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )
        # logging.info("word_predictions >>> " + str(word_predictions[masked_tgt_masks].detach().cpu().numpy()))
        # logging.info("tgt_tokens >>> " + str(tgt_tokens[masked_tgt_masks].detach().cpu().numpy()))

        # generate training labels for deletion
        if rd.random() > 0.2:
            word_del_targets = _get_del_targets(
                src_tokens, tgt_tokens, self.pad
            )
            # encoded_del = self.encoder(word_predictions, **kwargs)
            # encoded_del["encoder_out"] = encoded_del["encoder_out"][0].transpose(0, 1)
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                encoder_out=encoded_src,
            )
            # logging.info("word_del_targets >>> " + str(word_del_targets[0].cpu().numpy()))
            # logging.info("word_del_out >>> " + str(word_del_out[0].detach().argmax(-1).cpu().numpy()))
            word_del_masks = src_tokens.ne(self.pad)
        else:
            word_del_targets = _get_del_targets(
                word_predictions, tgt_tokens, self.pad
            )
            # logging.info("word_predictions >>> " + str(word_predictions[0].detach().cpu().numpy()))
            encoded_del = self.encoder(word_predictions, **kwargs)
            encoded_del["encoder_out"] = encoded_del["encoder_out"][0].transpose(0, 1)
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                encoder_out=encoded_del,
            )
            # logging.info("word_del_out >>> " + str(word_del_out[0].detach().argmax(-1).cpu().numpy()))
            word_del_masks = word_predictions.ne(self.pad)

        # assert mask_ins_out.size(1) == mask_ins_targets.size(1)
        # assert mask_ins_out.size(1) == mask_ins_masks.size(1)
        # assert word_ins_out.size(1) == tgt_tokens.size(1)
        # assert word_ins_out.size(1) == masked_tgt_masks.size(1)
        # assert word_del_out.size(1) == word_del_targets.size(1)
        # assert word_del_out.size(1) == word_del_masks.size(1)

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
                "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }

    def forward_encoder(
        self, input,
    ):
        encoder_out = self.encoder(
            input,
        )
        encoder_out["encoder_out"] = encoder_out["encoder_out"][0].transpose(0, 1)
        encoder_out["encoder_padding_mask"] = encoder_out["encoder_padding_mask"][0]
        return encoder_out

    def forward_decoder(
        self, decoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens  #  SEQ TO BE EDITED ?
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        max_lens = torch.zeros_like(output_tokens).fill_(255)[:, 0]

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            encoder_out = self.encoder(
                _skip(
                    output_tokens,
                    can_del_word),
                **kwargs)
            encoder_out["encoder_out"] = encoder_out["encoder_out"][0].transpose(0, 1)
            encoder_out["encoder_padding_mask"] = torch.stack(encoder_out["encoder_padding_mask"]).transpose(1, 2)
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                encoder_out=encoder_out,
            )
            word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(
                output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            encoder_out = self.encoder(
                _skip(
                    output_tokens,
                    can_ins_mask),
                **kwargs)
            encoder_out["encoder_out"] = encoder_out["encoder_out"][0].transpose(0, 1)
            encoder_out["encoder_padding_mask"] = torch.stack(encoder_out["encoder_padding_mask"]).transpose(1, 2)
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                encoder_out=encoder_out,
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(
                output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            encoder_out = self.encoder(
                _skip(
                    output_tokens,
                    can_ins_mask),
                **kwargs)
            encoder_out["encoder_out"] = encoder_out["encoder_out"][0].transpose(0, 1)
            encoder_out["encoder_padding_mask"] = torch.stack(encoder_out["encoder_padding_mask"]).transpose(1, 2)
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                encoder_out=encoder_out,
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(
                output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        # initial_output_tokens[:, 0] = self.bos
        # initial_output_tokens[:, 1] = self.eos

        # initial_output_scores = initial_output_tokens.new_zeros(
        #     *initial_output_tokens.size()
        # ).type_as(encoder_out["encoder_out"][0])

        initial_output_scores = src_tokens.new_zeros(
            *src_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=src_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        # cfg = TransformerConfig.from_namespace(args)
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        # if path:
        #     embed_dict = utils.parse_embedding(path)
        #     utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoderBase(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = CorrectionLevenshteinTransformerDecoder(cfg, tgt_dict, embed_tokens)
        return decoder


class CorrectionLevenshteinTransformerDecoder(FairseqDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(
            dictionary
        )
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)
        self.ensemble_models = None

        # self.dropout_module = FairseqDropout(
        #     cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        # )
        # self.decoder_layerdrop = cfg.decoder.layerdrop
        # self.share_input_output_embed = cfg.share_decoder_input_output_embed

        # input_embed_dim = embed_tokens.embedding_dim
        # embed_dim = cfg.decoder.embed_dim
        # self.embed_dim = embed_dim
        self.output_embed_dim = embed_tokens.embedding_dim
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        # self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        # self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        # self.cross_self_attention = cfg.cross_self_attention

        # self.build_output_projection(cfg, dictionary, embed_tokens)

        # AFTER
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(
            cfg, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        # self.early_exit = [int(i) for i in args.early_exit.split(",")]
        # assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        # self.layers_msk = None
        # if getattr(args, "no_share_maskpredictor", False):
        #     self.layers_msk = nn.ModuleList(
        #         [
        #             TransformerDecoderLayer(args, no_encoder_attn)
        #             for _ in range(self.early_exit[1])
        #         ]
        #     )
        # self.layers_del = None
        # if getattr(args, "no_share_discriminator", False):
        #     self.layers_del = nn.ModuleList(
        #         [
        #             TransformerDecoderLayer(args, no_encoder_attn)
        #             for _ in range(self.early_exit[0])
        #         ]
        #     )

        if getattr(cfg, "share_discriminator_maskpredictor", False):
            assert getattr(
                cfg, "no_share_discriminator", False
            ), "must set saperate discriminator"
            self.layers_msk = self.layers_del
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        return prev_output_tokens, {"attn": None, "inner_states": None}

    def build_output_projection(self, args, dictionary, embed_tokens):
        cfg = TransformerConfig.from_namespace(args)
        self.output_projection = nn.Linear(
            self.output_embed_dim, len(dictionary), bias=False
        )
        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
            num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    @ensemble_decoder
    def forward_mask_ins(
            self,
            normalize,
            encoder_out,
            **unused):
        # features, extra = self.extract_features(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     early_exit=self.early_exit[1],
        #     layers=self.layers_msk,
        #     **unused
        # )
        # logging.info(str(encoder_out["encoder_out"][0].shape))
        features_cat = torch.cat([encoder_out["encoder_out"][:, :-1, :],
                                  encoder_out["encoder_out"][:, 1:, :]],
                                  2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), encoder_out["encoder_padding_mask"]
        return decoder_out, encoder_out["encoder_padding_mask"]

    @ensemble_decoder
    def forward_word_ins(
            self,
            normalize,
            encoder_out,
            **unused):
        # features, extra = self.extract_features(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     early_exit=self.early_exit[2],
        #     layers=self.layers,
        #     **unused
        # )
        decoder_out = self.output_layer(encoder_out["encoder_out"])
        if normalize:
            return F.log_softmax(decoder_out, -1), encoder_out["encoder_padding_mask"]
        return decoder_out, encoder_out["encoder_padding_mask"]

    @ensemble_decoder
    def forward_word_del(
            self,
            normalize,
            encoder_out,
            **unused):
        # features, extra = self.extract_features(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     early_exit=self.early_exit[0],
        #     layers=self.layers_del,
        #     **unused
        # )
        decoder_out = F.linear(encoder_out["encoder_out"], self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), encoder_out["encoder_padding_mask"]
        return decoder_out, encoder_out["encoder_padding_mask"]


@register_model_architecture("correction_levenshtein_transformer",
                             "correction_levenshtein_transformer")
def correction_levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(
        args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(
        args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False
    )
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)


@register_model_architecture("correction_levenshtein_transformer",
                             "correction_levenshtein_transformer_wmt_en_de")
def correction_levenshtein_transformer_wmt_en_de(args):
    correction_levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper
# (Vaswani et al., 2017)
@register_model_architecture("correction_levenshtein_transformer",
                             "correction_evenshtein_transformer_vaswani_wmt_en_de_big")
def correction_levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    correction_levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("correction_levenshtein_transformer",
                             "correction_levenshtein_transformer_wmt_en_de_big")
def correction_levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    correction_levenshtein_transformer_vaswani_wmt_en_de_big(args)
