# OT-Seq2Seq-PyTorch

This repository is an unofficial PyTorch implementation of CLR 2019 paper [IMPROVING SEQUENCE-TO-SEQUENCE LEARNING
VIA OPTIMAL TRANSPORT](https://arxiv.org/pdf/1901.06283.pdf)

Official TensorFlow implementation is [here](https://github.com/LiqunChen0606/OT-Seq2Seq).

This repository uses [Weights and Biases](https://wandb.ai/site).

### Environment

#### Docker
```bash
docker build -t ot-seq2seq-pytorch docker/
docker run --rm -it --gpus device=0 -v $PWD:/work -w /work ot-seq2seq-pytorch bash
```
or simply install requirements:
```bash
python -m pip install -r docker/requirements.txt
```