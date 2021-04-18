from collections import defaultdict

from features.feature_utils import build_index_by_count


class FeatureMultiLabel:
    def __init__(self, multi_label_separator=' '):
        self.multi_label_separator = multi_label_separator
        self.__value2stat = defaultdict(lambda: defaultdict(int))
        self.__value2id = defaultdict(int)

    def __len__(self):
        return len(self.__value2id)

    def update(self, value):
        for label in value.split(self.multi_label_separator):
            self.__value2stat[label]['count'] += 1

    def handle(self, batch_value):
        label_vec = []
        for multi_label in batch_value:
            one_hot = [0.] * len(self.__value2id)
            for label in multi_label.split(self.multi_label_separator):
                one_hot[self.__value2id[label]] = 1.
            label_vec.append(one_hot)
        return label_vec

    def build_index(self):
        self.__value2stat = build_index_by_count(self.__value2stat, 0, offset=0)
        self.__value2id = {x: y['index'] for x, y in self.__value2stat.items()}

    def save(self):
        return self.__value2stat

    def load(self, value2stat):
        self.__value2stat = value2stat
        self.__value2id = {x: y['index'] for x, y in value2stat.items()}

