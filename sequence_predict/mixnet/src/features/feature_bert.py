from pytorch_pretrained_bert import BertTokenizer
import torch


class FeatureBert:
    def __init__(self, config):
        self.model_dir = config["model_dir"]
        self.cache_dir = config["cache_dir"]
        self.do_lower_case = config["do_lower_case"]
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir,
                                                       cache_dir=self.cache_dir,
                                                       do_lower_case=self.do_lower_case)
        self.token_length = config.get('bert_length', 128)

    def __len__(self):
        return 0

    def update(self, text):
        pass

    def handle(self, text_feature_list):
        batch_input_ids = []
        batch_input_mask = []
        batch_segment_ids = []
        for feature in text_feature_list:
            tokens = self.tokenizer.tokenize(feature)
            if len(tokens) > self.token_length - 2:
                tokens = tokens[0:self.token_length - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Zero-pad up to the sequence length.
            padding = [0] * (self.token_length - len(input_ids))
            masked_ids = [1] * len(input_ids) + padding
            input_ids += padding
            segments_ids = [0] * self.token_length
            # Add to batch.
            batch_input_ids.append(input_ids)
            batch_input_mask.append(masked_ids)
            batch_segment_ids.append(segments_ids)
        batch_vec = {
            'inputs': torch.tensor(batch_input_ids, dtype=torch.long),
            'segments': torch.tensor(batch_input_mask, dtype=torch.long),
            'masks': torch.tensor(batch_segment_ids, dtype=torch.long)
        }
        return batch_vec

    def build_index(self):
        pass

    def save(self):
        return {}

    def load(self, value2stat):
        pass

    def info(self):
        return {
            "type": "bert",
            "token_count": 0,
            "embedding_size": 0
        }
