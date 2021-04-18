import torch
import torch.nn as nn


class EmbedLayers(nn.Module):
    """Embedding layers. """

    def __init__(self, embedding_info, ner_model=None):
        super(EmbedLayers, self).__init__()
        self.embedding_info = embedding_info
        self.ner_model = ner_model
        # Init Embeddings.
        self.embedding_layer_dict = nn.ModuleDict()
        for fea_name, emb_info in self.embedding_info.items():
            # TODO: add pretrained embedding.
            if emb_info['type'] == 'value':
                enum_name = f'{fea_name}_enum'
                enum_embedding_list = nn.ModuleList()
                for enum_info in emb_info['enum_info']:
                    # TODO: Check whether adding padding_idx=0.
                    enum_embedding_list.append(nn.Embedding(enum_info['token_count'],
                                                            enum_info['embedding_size'],
                                                            padding_idx=0))
                self.embedding_layer_dict[enum_name] = enum_embedding_list
            elif emb_info['type'] == 'vector':
                continue
            elif emb_info['token_count'] > 0:
                self.embedding_layer_dict[fea_name] = nn.Embedding(emb_info['token_count'],
                                                                   emb_info['embedding_size'],
                                                                   padding_idx=0)

    def forward(self, inputs):
        inputs_embedding = {}
        seq_length = {}

        for fea_name, emb_info in self.embedding_info.items():
            if emb_info['type'] == 'value':
                enum_name = f'{fea_name}_enum'
                enum_inputs = inputs[fea_name]['enum']
                batch_size, enum_length = enum_inputs.shape
                enum_embedding_list = []
                for i in range(enum_length):
                    # Get embedding for each value.
                    enum_embedding_list.append(self.embedding_layer_dict[enum_name][i](enum_inputs[:, i]))
                # value_embedding: [batch_size, total_embedding_size]
                if len(enum_embedding_list) == 0:
                    enum_embedding = enum_inputs.new_tensor([], dtype=torch.float32)
                else:
                    enum_embedding = torch.cat(enum_embedding_list, dim=1)
                inputs_embedding[enum_name] = enum_embedding
                # seq_length[name]: [batch_size]
                seq_length[enum_name] = enum_inputs.new_full((batch_size,), enum_length)
            elif emb_info['type'] in ['bert']:
                continue
            elif emb_info['type'] in ['vector']:
                inputs_embedding[fea_name] = inputs[fea_name] 
            else:
                # embedding_x[name]: [batch_size, max_seq_length, embedding_size]
                inputs_embedding[fea_name] = self.embedding_layer_dict[fea_name](inputs[fea_name])
                # seq_length[name]: [batch_size]
                seq_length[fea_name] = self.__get_seq_length(inputs[fea_name])

        return inputs_embedding, seq_length

    def __get_seq_length(self, data):
        # mask: [batch_size x max_seq_length]
        mask = torch.gt(data, 0).long()
        # seq_length: [batch_size]
        seq_length = torch.sum(mask, dim=1)
        return seq_length
