from collections import defaultdict
import os
import torch

from features.feature_utils import build_index_by_count


class FeatureValue:
    def __init__(self, config):
        self.value_separator = config.get('value_seperator', ' ')
        self.valid_key_file = config.get('valid_key_file', None)
        self.min_count = config.get('min_count', 10)
        if self.valid_key_file is not None:
            self.value_valid_keys = FeatureValue.load_valid_key_file(self.valid_key_file)
        else:
            self.value_valid_keys = None

        self.__enum2stat = defaultdict(lambda: {"count": 0,
                                                "values": defaultdict(lambda: {"count": 0})})
        self.__value2stat = defaultdict(lambda: {'count': 0})
        self.__value2id = defaultdict(int)
        self.__enum2id = defaultdict(int)

    def __len__(self):
        return len(self.__value2id)

    def update(self, value_str):
        # TODO: check this logic that I've changed.
        for w in value_str.split(self.value_separator):
            if not w:
                continue
            sub_arr = w.split(':', 1)
            if len(sub_arr) == 2:
                key, value = sub_arr
                if key.startswith('__fac__'):
                    key = key[7:]
                # Get key type.
                key_type = 'ignore_type'
                if self.value_valid_keys is None:
                    key_type = 'enum_type'
                else:
                    key_type = self.value_valid_keys.get(key, 'ignore_type')

                # Update feature stat.
                if key_type == 'enum_type':
                    # enum_name = f'{feature_name}_enum'
                    self.__enum2stat[key]['count'] += 1
                    self.__enum2stat[key]['values'][value]['count'] += 1
                elif key_type == 'value_type':
                    # value_name = f'{feature_name}_value'
                    self.__value2stat[key]['count'] += 1
                    try:
                        float_value = float(value)
                        value_stat = self.__value2stat[key]
                        if 'min' not in value_stat:
                            value_stat['min'] = float_value
                            value_stat['max'] = float_value
                        elif float_value < value_stat['min']:
                            value_stat['min'] = float_value
                        elif float_value > value_stat['max']:
                            value_stat['max'] = float_value
                    except ValueError:
                        continue
            else:
                raise ValueError(f'The value {w} is not with "key:value" format.')

    def info(self):
        enum_info_list = []
        enum_embedding_size = 0
        for key, value_dict in self.__enum2stat.items():
            count = len(value_dict['values']) + 1
            embedding_size = round(pow(count, 0.25) * 6)
            enum_embedding_size += embedding_size
            enum_info_list.append({
                'token_count': len(value_dict['values']) + 1,
                'embedding_size': embedding_size
            })
        return {
            "type": "value",
            'embedding_size': enum_embedding_size + len(self.__value2id),
            'enum_info': enum_info_list
        }

    def handle(self, value_feature_list):
        batch_enum_vec = []
        batch_value_vec = []
        for feature in value_feature_list:
            enum_vec = [0] * len(self.__enum2stat)
            value_vec = [-1.] * len(self.__value2id)
            if not feature:
                batch_enum_vec.append(enum_vec)
                batch_value_vec.append(value_vec)
                continue
            value_items = feature.split(self.value_separator)
            for item in value_items:
                sub_arr = item.split(':', 1)
                if len(sub_arr) == 2:
                    key, value = sub_arr
                    if key.startswith('__fac__'):
                        key = key[7:]
                    if key in self.__enum2id:
                        key_idx = self.__enum2id[key]['index']
                        val_idx = self.__enum2id[key]['values'].get(value, 0)
                        enum_vec[key_idx] = val_idx
                    elif key in self.__value2id:
                        key_idx = self.__value2id[key]['index']
                        min_value = self.__value2id[key].get('min', 0.)
                        max_value = self.__value2id[key].get('max', 0.)
                        value_vec[key_idx] = self.format_value(value, min_value, max_value)
            batch_enum_vec.append(enum_vec)
            batch_value_vec.append(value_vec)
        return {
            "enum": torch.tensor(batch_enum_vec),
            "value": torch.tensor(batch_value_vec)
        }

    def build_index(self):
        for _, item_dict in self.__enum2stat.items():
            item_value_dict = item_dict["values"]
            item_dict['values'] = build_index_by_count(item_value_dict, self.min_count, offset=1)

        self.__value2stat = build_index_by_count(self.__value2stat, self.min_count, offset=0)
        self.__value2id = self.__value2stat# {x: y['index'] for x, y in self.__value2stat.items()}

    def save(self):
        return {
            "value": self.__value2stat,
            "enum": self.__enum2stat
        }

    def load(self, value2stat):
        self.__value2stat = value2stat["value"]
        self.__enum2stat = value2stat["enum"]
        self.__value2id = self.__value2stat# self.__value2id = {x: y['index'] for x, y in self.__value2stat}

    @staticmethod
    def format_value(value, min_value, max_value):
        """Value normalization."""
        try:
            float_value = float(value)
            range_value = max_value - min_value
            if range_value == 0:
                format_value = 1.
            else:
                format_value = (float_value - min_value) / range_value
        except ValueError:
            format_value = 0.
        return format_value

    @staticmethod
    def load_valid_key_file(file_path):
        """Loading valid keys from file."""
        value_valid_keys = {}
        assert os.path.exists(file_path)
        with open(file_path, 'r', encoding='utf-8') as in_file:
            for line in in_file:
                line = line.rstrip()
                if line == '':
                    continue
                tokens = line.rstrip().split('\t')
                if len(tokens) == 2:
                    value_valid_keys[tokens[0]] = tokens[1].lower()
                elif len(tokens) == 1:
                    value_valid_keys[tokens[0]] = 'enum_type'
                else:
                    raise NotImplementedError('invalid key file:' + line)
        return value_valid_keys
