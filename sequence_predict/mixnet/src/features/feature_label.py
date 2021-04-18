from collections import defaultdict

from features.feature_utils import build_index_by_count


class FeatureLabel:
    def __init__(self, multi_turn_separator='→', mask='-1'):
        self.multi_turn_separator = multi_turn_separator
        self.mask_label = mask
        self.__value2stat = defaultdict(lambda: defaultdict(int))
        self.__value2id = defaultdict(int)

    def __len__(self):
        return len(self.__value2id)

    def update(self, value):
        label_list = value.split(self.multi_turn_separator)
        for label in label_list:
            if self.mask_label is not None and label == self.mask_label:
                continue
            self.__value2stat[label]['count'] += 1

    def handle(self, batch_value):
        label_list = []
        for labels in batch_value:
            label_list.extend(labels.split(self.multi_turn_separator))

        label_vec = []
        for label in label_list:
            if self.mask_label is not None and label == self.mask_label:
                label_vec.append(-1)
            else:
                label_vec.append(self.__value2id[label])
        return label_vec

    def build_index(self):
        self.__value2stat = build_index_by_count(self.__value2stat, 0, offset=0)
        self.__value2id = {x: y['index'] for x, y in self.__value2stat.items()}

    def save(self):
        return self.__value2stat

    def load(self, value2stat):
        self.__value2stat = value2stat
        self.__value2id = {x: y['index'] for x, y in value2stat.items()}

