<!-- MIT License

Copyright (c) 2019 LiqunChen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. -->

This code is largely borrowed from [texar-pytorch](https://github.com/asyml/texar-pytorch).
Please install the requirement first to run the code.

# Sequence Generation #
This example provide implementations of some classic and advanced training algorithms that tackles the exposure bias. The base model is an attentional seq2seq.

* **Maximum Likelihood (MLE)**: attentional seq2seq model with maximum likelihood training.
* **Maximum Likelihood (MLE) + Optimal transport (OT)**: Described in [OT-seq2seq](https://arxiv.org/pdf/1901.06283.pdf) and we use the sampling approach (n-gram replacement) by [(Ma et al., 2017)](https://arxiv.org/abs/1705.07136).

## Usage ##

### Dataset ###

Two example datasets are provided:

  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset, following [(Ranzato et al., 2015)](https://arxiv.org/pdf/1511.06732.pdf) for data pre-processing.
  * gigaword: The benchmark [GIGAWORD](https://catalog.ldc.upenn.edu/LDC2003T05) text summarization dataset. we sampled 200K out of the 3.8M pre-processed training examples provided by [(Rush et al., 2015)](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf) for the sake of training efficiency. We used the refined validation and test sets provided by [(Zhou et al., 2017)](https://arxiv.org/pdf/1704.07073.pdf).

Download the data with the following commands:

```
python utils/prepare_data.py --data iwslt14
python utils/prepare_data.py --data giga
```

### Train the models ###

#### Baseline Attentional Seq2seq with OT

```
python baseline_seq2seq_attn_main.py \
    [--config-model configs.config_model] \
    [--config-data configs.config_iwslt14] \
    [--seed 0]\
    [--project-name OT-Seq2Seq-PyTorch]\
    [--log-path None]
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix. Default value is `configs.config_model`.
  * `--config_data` specifies the data config. Note not to include the `.py` suffix. Default value is `configs.config_iwslt14`.
  * `--seed` specifies a random seed. Default value is `0`.
  * `--project-name` specifies wandb project name. Default value is `OT-Seq2Seq-PyTorch`.
  * `--log-path` specifies the output log file path. Default value is `None`.

[configs.config_model.py](./configs/config_model.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder. Hyperparameters taking default values can be omitted from the config file.

## Results ##

### Machine Translation
| Model      | BLEU Score   |
| -----------| -------|
| MLE        | 25.42 |
| MLE + OT |  25.95 |

### Text Summarization (F1 score)
| Model      | Rouge-1   | Rouge-2 | Rouge-L |
| -----------| -------|-------|-------|
| MLE        | 34.64  | 15.94 | 30.93 |
| MLE + OT | 35.23 | 16.24 | 31.44 |