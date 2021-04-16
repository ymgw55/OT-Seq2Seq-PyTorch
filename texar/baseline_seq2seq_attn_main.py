# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attentional Seq2seq.
"""

import argparse
import importlib
import logging
import random
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn

import texar.torch as tx
from rouge import Rouge
import wandb


def seed_torch(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_path=None,
               log_level=logging.INFO) -> logging.Logger:

    logger = logging.getLogger()
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, 'w'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-model', type=str, default="configs.config_model",
        help="The model config.")
    parser.add_argument(
        '--config-data', type=str, default="configs.config_iwslt14",
        help="The dataset config.")
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed.'
    )
    parser.add_argument(
        '--project-name', type=str, default="OT-Seq2Seq-PyTorch",
        help="wandb project name.")
    parser.add_argument(
        '--log-path', type=str, default=None,
        help="Output log path.")
    args = parser.parse_args()
    return args


class Seq2SeqAttn(nn.Module):

    def __init__(self,
                 train_data: tx.data.data.paired_text_data.PairedTextData,
                 config_model: Any, device: str) -> None:
        super().__init__()

        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=config_model.embedder)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=config_model.embedder)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=self.source_embedder.dim,
            hparams=config_model.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=self.target_embedder.dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)

        self.beam_width = config_model.beam_width
        self.device = device

    def forward(self,
                batch: tx.data.data.dataset_utils.Batch,
                mode: str):

        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch['source_text_ids']),
            sequence_length=batch['source_length'])

        memory = torch.cat(enc_outputs, dim=2)

        if mode == "train":
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")

            training_outputs, _, _ = self.decoder(
                memory=memory,
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=batch['target_text_ids'][:, :-1],
                sequence_length=batch['target_length'] - 1)

            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['target_text_ids'][:, 1:],
                logits=training_outputs.logits,
                sequence_length=batch['target_length'] - 1)

            return mle_loss
        else:
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)

            infer_outputs = self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch['source_length'],
                beam_width=self.beam_width)

            return infer_outputs


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    project_name = f'{args.project_name}'
    data_name = Path(args.config_data).suffix[1:]
    exp_name = 'baseline'
    if 'iwslt14' in data_name:
        exp_name += '/iwslt14'
    elif 'giga' in data_name:
        exp_name += '/giga'

    config_data: Any = importlib.import_module(args.config_data)
    tags = [config_data.eval_metric, 'baseline']

    wandb.init(project=project_name,
               name=exp_name,
               tags=tags)

    seed = args.seed
    log_path = args.log_path

    logger = get_logger(log_path)
    logger.info(f'seed: {seed}')
    seed_torch(seed)

    train_data = tx.data.PairedTextData(
        hparams=config_data.train, device=device)
    val_data = tx.data.PairedTextData(
        hparams=config_data.val, device=device)
    test_data = tx.data.PairedTextData(
        hparams=config_data.test, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    config_model: Any = importlib.import_module(args.config_model)
    model = Seq2SeqAttn(train_data, config_model, device)
    model.to(device)
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_model.opt)

    def _train_epoch(epoch: int) -> None:
        data_iterator.switch_to_train_data()
        model.train()

        step = 0
        for batch in data_iterator:
            loss = model(batch, mode="train")
            loss.backward()
            train_op()
            if step % config_data.display == 0:
                wandb.log({'train/train_loss': loss.item()})
                logger.info(f'epoch: {epoch}'
                            f' | step: {step} / {len(data_iterator._datasets["train"])}'
                            f' | loss: {loss:.4f}')
            step += 1

    @torch.no_grad()
    def _eval_epoch(mode: str) -> None:
        if mode == 'val':
            data_iterator.switch_to_val_data()
        else:
            data_iterator.switch_to_test_data()
        model.eval()

        refs, hypos = [], []
        for batch in data_iterator:
            infer_outputs = model(batch, mode="val")
            output_ids = infer_outputs["sample_id"][:, :, 0].cpu()
            target_texts_ori = [text[1:] for text in batch['target_text']]
            target_texts = tx.utils.strip_special_tokens(
                target_texts_ori, is_token_list=True)
            output_texts = tx.data.vocabulary.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            for hypo, ref in zip(output_texts, target_texts):
                if config_data.eval_metric == 'bleu':
                    hypos.append(hypo)
                    refs.append([ref])
                elif config_data.eval_metric == 'rouge':
                    hypos.append(tx.utils.compat_as_text(hypo))
                    refs.append(' '.join(tx.utils.compat_as_text(ref)))

        if config_data.eval_metric == 'bleu':
            return tx.evals.corpus_bleu_moses(
                list_of_references=refs, hypotheses=hypos)
        elif config_data.eval_metric == 'rouge':
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    def _calc_reward(score: Union[float, Dict]) -> float:
        """
        Return the bleu score or the sum of (Rouge-1, Rouge-2, Rouge-L).
        """
        if config_data.eval_metric == 'bleu':
            return score
        elif config_data.eval_metric == 'rouge':
            return 100*sum([value['f'] for value in score.values()])

    best_val_score = -1.
    for epoch in range(config_data.num_epochs):
        logger.info(f'Epoch: {epoch}')
        _train_epoch(epoch)

        val_score = _eval_epoch('val')
        test_score = _eval_epoch('test')

        best_val_score = max(best_val_score, _calc_reward(val_score))

        if config_data.eval_metric == 'bleu':
            wandb.log({'val/BLEU': val_score})
            wandb.log({'val/best_BLEU': best_val_score})

            logger.info(f'val  | epoch: {epoch}'
                  f' | BLEU: {val_score:.4f}'
                  f' | best-ever: {best_val_score:.4f}')

            wandb.log({'test/test_BLEU': test_score})
            logger.info(f'test | epoch: {epoch} | BLEU: {test_score:.4f}')

        else:
            wandb.log({'val/ROUGE_f1_sum': _calc_reward(val_score)})
            wandb.log({'val/best_ROUGE_f1_sum': best_val_score})
            logger.info(f'val  | epoch: {epoch}'
                  f' | ROUGE_f1_sum: {_calc_reward(val_score):.4f}'
                  f' | best-ever: {best_val_score:.4f}')

            fpr2name = {'f': 'f1_score', 'p': 'precision', 'r': 'recall'}
            for state, scores in [['val', val_score], ['test', test_score]]:
                logger.info(f'{state} rouge score:')
                for rouge_type, rouge_dict in scores.items():
                    for key, rouge_score in rouge_dict.items():
                        wandb.log({f'{state}/{rouge_type}/{fpr2name[key]}':
                                  100*rouge_score})
                        logger.info(f'{state}/{rouge_type}/{fpr2name[key]}: '
                                    f'{100*rouge_score:.4f}')

        logger.info('=' * 50)


if __name__ == '__main__':
    main()
