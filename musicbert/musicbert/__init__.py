# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import fairseq.tasks.sentence_prediction
import fairseq.tasks.masked_lm
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.data import (MaskTokensDataset,
                          LanguagePairDataset,
                          PrependTokenDataset,
                          data_utils)
                          
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import RobertaEncoder, RobertaModel
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import math
import logging
import os
import torch


logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256


# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_task("sentence_prediction_multilabel")
class MusicBERTSentencePredictionMultilabelTask(SentencePredictionTask):
    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, 'input0', split)
        input0 = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if self.args.init_token is not None:
            input0 = OctupleTokenDataset(input0)
        src_dataset = input0
        labels, label_lengths = [], []
        with open(os.path.join(self.args.data, 'label', split+".label")) as file:
            for line in file:
                line = line.strip()
                line = line.split()
                label = [self.label_dictionary.index(item) for item in line]

                if(len(label) < self.args.num_classes):
                    label = label + \
                        [self.label_dictionary.index(
                            '<pad>')]*(self.args.num_classes-len(label))

                label = label[:self.args.num_classes]

                label = torch.tensor(label)
                labels.append(label)
                label_lengths.append(len(label))
        assert len(src_dataset) == len(labels)
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.label_dictionary,
            tgt=labels,
            tgt_sizes=torch.tensor(label_lengths),
            tgt_dict=self.label_dictionary,
            left_pad_source=False,
            input_feeding=False,
        )


# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_criterion("sentence_prediction_multilabel")
class MusicBERTSentencePredictionMultilabelCriterion(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])
        targets = F.one_hot(targets.long(), num_classes=logits.size()[-1]+4)
        targets = targets.sum(dim=1)
        targets = targets[:, 4:]
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='sum')
        sample_size = logits.size()[0]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size * logits.size()[1],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = F.relu(torch.sign(logits))
        logging_output["ncorrect"] = sample_size - \
            torch.sign((preds != targets).sum(dim=1)).sum().data
        logging_output["y_true"] = targets.detach().cpu().numpy()
        logging_output["y_pred"] = torch.sigmoid(logits).detach().cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.vstack(tuple(log.get("y_pred")
                                     for log in logging_outputs if "y_pred" in log))
            y_true = np.vstack(tuple(log.get("y_true")
                                     for log in logging_outputs if "y_true" in log))
            for score in ["roc_auc_score", "f1_score"]:
                for average in ["macro", "micro", "weighted", "samples"]:
                    try:
                        y_score = np.round(
                            y_pred) if score == "f1_score" else y_pred
                        kwargs = {
                            "zero_division": 0} if score == "f1_score" else dict()
                        result = sklearn.metrics.__dict__[score](
                            y_true, y_score, average=average, **kwargs)
                        metrics.log_scalar(
                            "{}_{}".format(score, average), result)
                    except BaseException as e:
                        metrics.log_scalar(
                            "{}_{}".format(score, average), None)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False


class OctupleMaskTokensDataset(MaskTokensDataset):
    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)
            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )
            assert not self.mask_whole_words, 'mask whole words not supported for cp'

            def generate_mask(sz, prob):
                mask_n = np.random.rand(sz)
                mask_s = np.zeros(sz, dtype=np.int8)
                mask_s += mask_n < prob * \
                    (self.random_token_prob)  # 3 -> random
                mask_s += mask_n < prob * \
                    (self.random_token_prob +
                     self.leave_unmasked_prob)  # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s
            mask_prob = self.mask_prob
            mask = np.zeros_like(item, dtype=np.int8)
            # mask bos eos tokens (compound)
            mask[:8] = np.repeat(generate_mask(1, mask_prob), 8)
            # mask bos eos tokens (compound)
            mask[-8:] = np.repeat(generate_mask(1, mask_prob), 8)
            strategy = np.random.choice(mask_strategy)
            if strategy == 'element':  # element level mask
                mask[8: -8] = np.repeat(generate_mask(sz -
                                                      2 * 8, mask_prob), 1)
            if strategy == 'compound':  # compound token level mask
                mask[8: -8] = np.repeat(generate_mask(sz //
                                                      8 - 2, mask_prob), 8)
            if strategy == 'bar':  # bar level mask
                mask[8: -8] = generate_mask((max_bars * max_instruments + len(self.vocab)) * 8, mask_prob).reshape(-1, 8)[
                    ((item[8: -8: 8] - 4) * max_instruments) + (item[8 + 2: -8 + 2: 8] - 4)].flatten()
            if self.return_masked_tokens:
                new_item = item.numpy()[:]
                new_item[mask == 0] = self.pad_idx
                return torch.from_numpy(new_item)
            masked_item = np.random.choice(len(self.vocab), sz)
            set_original = np.isin(mask, [0, 2])
            masked_item[set_original] = item[set_original]
            set_mask = np.isin(mask, [1])
            masked_item[set_mask] = self.mask_idx
            return torch.from_numpy(masked_item)

#With reference from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_encoder.py
class OctupleEncoder(TransformerEncoder):
    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)
        self.tpu = False
        embedding_dim = embed_tokens.embedding_dim
        if not disable_cp:
            self.downsampling = nn.Sequential(
                nn.Linear(embedding_dim * 8, embedding_dim))
            self.upsampling = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 8))

    def forward_embedding(
        self, tokens, segment_labels , last_state_only, 
        positions ,token_embeddings: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        # if token_embedding is None:
        #     token_embedding = self.embed_tokens(src_tokens)
        # x = embed = self.embed_scale * token_embedding
        # if self.embed_positions is not None:
        #     x = embed + self.embed_positions(src_tokens)
        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)
        # x = self.dropout_module(x)
        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)
        # return x, embed

        ratio = 1 if disable_cp else 8
        if not disable_cp:
            assert tokens.shape[1] % ratio == 0, 'token sequences length should be multiple of ' + str(
                ratio) + ' for compound mode'
            assert last_state_only, 'hidden states not available for compound mode'
            assert positions is None, 'custom positions is not supported for compound mode'
            assert token_embeddings is None, 'custom token embeddings is not supported for compound mode'
            assert segment_labels is None, 'segment embedding not supported for compound mode'
        padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)
        if not disable_cp:
            x = self.downsampling(x.view(x.shape[0], x.shape[1] // ratio, -1))
        #self.embed_scale seems to have default value of 1 in fairseq v0.12.2:
        x = embed = self.embed_scale * token_embeddings
        if self.embed_positions is not None:
            x = x + \
                self.embed_positions(tokens[:, ::ratio], positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return x, embed, padding_mask


    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        segment_labels: torch.Tensor = None,
        positions: Optional[torch.Tensor] = None
    ):
        
        return self.forward_scriptable(
            src_tokens, src_lengths, last_state_only, token_embeddings,
            segment_labels , positions
        )

    def forward_scriptable(
        self,
        src_tokens: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        segment_labels: torch.Tensor = None,
        positions: Optional[torch.Tensor] = None,
    ):
        ratio = 1 if disable_cp else 8

        x, encoder_embedding, encoder_padding_mask = self.forward_embedding(src_tokens, segment_labels,
        last_state_only, positions ,token_embeddings)

        x = x.transpose(0, 1)

        # inner_states = []
        # if not last_state_only:
        #     inner_states.append(x)
        # for layer in self.layers:
        #     x, _ = layer(x, self_attn_padding_mask=padding_mask)
        #     if not last_state_only:
        #         inner_states.append(x)
        # if not disable_cp:
        #     x = x.transpose(0, 1)
        #     x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
        #     x = x.transpose(0, 1)
        # sentence_rep = x[0, :, :]
        # if last_state_only:
        #     inner_states = [x]
        # if self.traceable:
        #     return torch.stack(inner_states), sentence_rep
        # else:
        #     return inner_states, sentence_rep

        encoder_states = []
        fc_results = []
        
        return_all_hiddens = not last_state_only

        if return_all_hiddens:
            encoder_states.append(x)

        # nested tensor and BT enable
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False

        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True

                if int_version >= 1131 or (
                    int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):

            x = x.transpose(0, 1)

            if NT_version:
                if (
                    encoder_padding_mask is not None
                    and torch._nested_tensor_from_mask_left_aligned(
                        x, encoder_padding_mask.logical_not()
                    )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True

        # encoder layers
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask

        if encoder_padding_mask is not None:
            has_pads = True 
        else:
            has_pads = False
        
        encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            lr = layer(x, encoder_padding_mask=encoder_padding_mask_out)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        #Upsampling octuples back to initial shape after forwarding through encoder layers
        if not disable_cp:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)


        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        # IMPORTANT!!: Layernorm at this step was not present in original OctupleEncoder script w/ fairseq v0.10.1, hence commenting it out from `TransformerEncoder` code
        # See usage of `cfg.layernorm_embedding` and `cfg.encoder.normalize_before` in https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_encoder.py#L77
        
        # if self.layer_norm is not None:
        #     x = self.layer_norm(x)


        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }



class MusicBERTEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, self.embed_tokens)

        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = OctupleEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder


@register_model("musicbert")
class MusicBERTModel(RobertaModel):
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        encoder = MusicBERTEncoder(args, task.source_dictionary)
        return cls(args, encoder)


@register_model_architecture("musicbert", "musicbert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("musicbert", "musicbert_base")
def musicbert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_large")
def musicbert_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_medium")
def musicbert_medium_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_small")
def musicbert_small_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_mini")
def musicbert_mini_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_tiny")
def musicbert_tiny_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    base_architecture(args)


class OctupleTokenDataset(PrependTokenDataset):
    def adaptor(self, e):
        prev_bar = None
        prev_pos = None
        prev_prog = None
        new_e = []
        for i in e:
            if prev_bar != i[0]:
                prev_bar = i[0]
                prev_pos = None
                new_e.append((i[0], None, None, None, None, None, i[6], None))
            if prev_pos != i[1]:
                prev_pos = i[1]
                prev_prog = None
                new_e.append((None, i[1], None, None, None, None, None, i[7]))
            if prev_prog != i[2]:
                prev_prog = i[2]
                new_e.append((None, None, i[2], None, None, None, None, None))
            if True:
                new_e.append((None, None, None, i[3], i[4], i[5], None, None))
        return new_e

    def convert(self, item):
        encoding = item[8: -8].tolist()
        encoding = list(tuple(encoding[i: i + 8])
                        for i in range(0, len(encoding), 8))
        encoding = self.adaptor(encoding)
        if convert_encoding == 'CP':
            encoding = list(3 if j is None else j for i in encoding for j in i)[
                :crop_length * 8]
        elif convert_encoding == 'REMI':
            encoding = list(j for i in encoding for j in i if j is not None)[
                :crop_length]
        else:
            assert False, 'Unknown encoding format'
        bos = 0
        eos = 2
        encoding = ([bos] * 8) + encoding + ([eos] * 8)
        return torch.tensor(encoding)

    def __init__(self, dataset, token=None):
        super().__init__(dataset, token=None)
        if convert_encoding != 'OCTMIDI':
            self._sizes = np.array([len(self.convert(i)) for i in dataset])
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if convert_encoding != 'OCTMIDI':
            item = self.convert(item)
        return item

    def num_tokens(self, index):
        return self._sizes[index].item()

    def size(self, index):
        return self._sizes[index].item()


fairseq.tasks.sentence_prediction.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.MaskTokensDataset = OctupleMaskTokensDataset
