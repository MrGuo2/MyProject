import torch


class FeatureVector:
    def __init__(self, config):
        self.length = config.get('length', ' ')
        self.value_separator = config.get('value_seperator', ' ')

    def __len__(self):
        return self.length

    def update(self, value_str):
        for w in value_str.split(self.value_separator):
            try:
                sub_arr = w.split(':')
                assert len(sub_arr) == 2
                assert 0 <= int(sub_arr[0]) < self.length
                float_value = float(sub_arr[1])
            except:
                raise ValueError(f'The value {w} is not with "key:value" format.')

    def info(self):
        return {
            "type": "vector",
            "token_count": self.length,
            "embedding_size": self.length
        }

    def handle(self, value_feature_list):
        batch_value_vec = []
        for feature in value_feature_list:
            value_vec = [0.] * self.length
            value_items = feature.split(self.value_separator)
            for item in value_items:
                sub_arr = item.split(':')
                if len(sub_arr) == 2:
                    try:
                        idx = int(sub_arr[0])
                        value = float(sub_arr[1])
                        value_vec[idx] = value
                    except:
                        continue
            batch_value_vec.append(value_vec)
        return torch.tensor(batch_value_vec)

    def build_index(self):
        pass

    def save(self):
        return {}

    def load(self, value2stat):
        pass
