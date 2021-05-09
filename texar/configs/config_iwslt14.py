# MIT License

# Copyright (c) 2019 LiqunChen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

num_epochs = 10 # the best epoch occurs within 10 epochs in most cases
display = 100

eval_metric = 'bleu'

batch_size = 128
source_vocab_file = './data/iwslt14/vocab.de'
target_vocab_file = './data/iwslt14/vocab.en'

train = {
    'batch_size': batch_size,
    'shuffle': True,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14/train.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/train.en',
        'vocab_file': target_vocab_file,
    }
}
val = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/iwslt14/valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/valid.en',
        'vocab_file': target_vocab_file,
    }
}
test = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/iwslt14/test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/test.en',
        'vocab_file': target_vocab_file,
    }
}
