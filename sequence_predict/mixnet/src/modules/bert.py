from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn


class BertNet(nn.Module):
    def __init__(self, params):
        super(BertNet, self).__init__()
        self.bert_model_dir = params.get('model_dir', 'bert-base-chinese')
        self.dropout_rate = params.get('dropout_rate', 0.4)
        self.cache_dir = params.get('cache_dir', './pytorch_pretrained_bert')
        self.bert_net = BertModel.from_pretrained(self.bert_model_dir, cache_dir=self.cache_dir)
        self.dropout = nn.Dropout(self.dropout_rate)

    @property
    def output_size(self):
        return self.bert_net.config.hidden_size

    def forward(self, data_dict):
        input_ids, token_type_ids, attention_mask = data_dict['inputs'], data_dict['segments'], data_dict['masks']
        _, pooled_output = self.bert_net(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        outputs = self.dropout(pooled_output)
        return outputs
