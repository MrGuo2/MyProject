from collections import defaultdict
import numpy as np
import torch

from features.feature_utils import build_index_by_count, pad_vec


class FeatureSeq:
    def __init__(self, config):
        self.seq_separator = config.get('seperator', ' ')
        self.token_length = config.get('length', 200)
        self.min_count = config.get('min', 1)
        self.embedding_dim = config.get('embedding_dim', 50)

        self.__value2stat = defaultdict(lambda: defaultdict(int))
        self.__value2id = defaultdict(int)

    def __len__(self):
        return len(self.__value2stat)

    def update(self, seq_str):
        seq_list = seq_str.split(self.seq_separator) + ['<eos>']
        for w in seq_list:
            if w:
                self.__value2stat[w]['count'] += 1

    def handle(self, batch_seq):
        batch_vec = []
        for seq in batch_seq:
            # Add ['<eos>'] to ensure length > 0.
            seq_list = seq.split(self.seq_separator) + ['<eos>']
            vec = [self.__value2id[w] for w in seq_list if w in self.__value2id]
            vec = pad_vec(np.array(vec, dtype=np.int64), self.token_length, padding_value=0)
            batch_vec.append(vec)
        return torch.tensor(batch_vec)

    def build_index(self):
        self.__value2stat = build_index_by_count(self.__value2stat, self.min_count, offset=1)
        self.__value2id = {x: y['index'] for x, y in self.__value2stat.items()}

    def save(self):
        return self.__value2stat

    def load(self, value2stat):
        self.__value2stat = value2stat
        self.__value2id = {x: y['index'] for x, y in value2stat.items()}

    def info(self):
        return {
            'type': 'seq',
            'token_count': len(self.__value2id) + 1,
            'embedding_size': self.embedding_dim
        }
