from collections import defaultdict
import numpy as np
import torch

from features.feature_utils import build_index_by_count, pad_vec


class FeatureText:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.seg_type = config.get('seg_type', 'word')
        self.seq_seperator = config.get('seperator', ' ')
        self.ngram = config.get('ngram', 1)
        self.token_length = config.get('length', 200)
        self.embedding_dim = config.get('embedding_dim', 50)
        self.min_count = config.get('min', 20)
        self.remove_stopwords = config.get('remove_stopwords', False)

        self.__value2stat = defaultdict(lambda: defaultdict(int))
        self.value2id = defaultdict(int)

    def __len__(self):
        return len(self.__value2stat)

    def update(self, text):
        self.update_list([text])

    def update_list(self, text_list):
        for sentence in text_list:
            tokens = self.tokenizer.tokenizer(sentence, seg_type=self.seg_type, ngram=self.ngram,
                                              remove_stopwords=self.remove_stopwords)
            tokens = tokens + ['<eos>']
            for w in tokens:
                if w:
                    self.__value2stat[w]['count'] += 1

    def handle(self, text_feature_list):
        return self.handle_list(text_feature_list)

    def handle_list(self, text_feature_list):
        batch_vec = []
        for feature in text_feature_list:
            tokens = self.tokenizer.tokenizer(feature,
                                              seg_type=self.seg_type,
                                              ngram=self.ngram,
                                              remove_stopwords=self.remove_stopwords)
            # Add ['<eos>'] to ensure length > 0.
            tokens = tokens + ['<eos>']
            vec = [self.value2id[w] for w in tokens if w in self.value2id]
            vec = pad_vec(np.array(vec, dtype=np.int64), self.token_length, padding_value=0)
            batch_vec.append(vec)
        return torch.tensor(batch_vec)

    def build_index(self):
        self.__value2stat = build_index_by_count(self.__value2stat, self.min_count, offset=1)
        self.value2id = {x: y['index'] for x, y in self.__value2stat.items()}

    def save(self):
        return self.__value2stat

    def load(self, value2stat):
        self.__value2stat = value2stat
        self.value2id = {x: y['index'] for x, y in value2stat.items()}

    def info(self):
        return {
            'type': 'text',
            'token_count': len(self.value2id) + 1,
            'embedding_size': self.embedding_dim
        }


class FeatureMultiTurnText(FeatureText):
    def __init__(self, config, tokenizer):
        super(FeatureMultiTurnText, self).__init__(config, tokenizer)
        self.multi_turn_separator = config.get('multi_turn_separator', '→')

    # def __update_list(self, text_list):
    #     super(FeatureMultiTurnText, self).__update_list(text_list)

    def update(self, text):
        text_list = text.split(self.multi_turn_separator)
        self.update_list(text_list)

    def handle(self, text_feature_list):
        text_list = []
        for multi_turn_text in text_feature_list:
            text_list.extend(multi_turn_text.split(self.multi_turn_separator))
        return self.handle_list(text_list)

    def info(self):
        return {
            'type': 'multi_turn_text',
            'token_count': len(self.value2id) + 1,
            'embedding_size': self.embedding_dim
        }
