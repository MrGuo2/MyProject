from collections import OrderedDict

import torch
import torch.nn as nn

from modules import *
from utils.log import logger
from collections import Counter, defaultdict


class MixNet(nn.Module):

    def __init__(self, model_config_dict, output_config_dict, embedding_dict, label_info_dict, ner_model=None):
        super(MixNet, self).__init__()
        self.embedding_dict = embedding_dict
        self.label_info_dict = label_info_dict
        # Use OrderedDict to keep order. But in Python 3.6+, dict are ordered.
        self.model_config_dict = OrderedDict(model_config_dict)
        self.multi_turn = False
        self.ner_model = ner_model
        self.output_config_dict = output_config_dict
        # Create embedding layers.
        self.embedding_layers = EmbedLayers(self.embedding_dict, self.ner_model)
        # Create subnets.
        self.subnet_dict = nn.ModuleDict()
        self.total_subnet_output_length = Counter()
        for name, config in self.model_config_dict.items():
            subnet = self.__get_subnet(config)
            subnet_output_name = config.get('output')
            logger.info(f'Creating {name} done.')
            self.subnet_dict[name] = subnet
            for output_name, output_info in self.output_config_dict.items():
                if (not output_info.get('input')) or (subnet_output_name in output_info.get('input', [])):
                    self.total_subnet_output_length[output_name] += subnet.output_size
        self.total_subnet_output_length = dict(self.total_subnet_output_length)
        # Last output mapping.
        self.output_layer_dict = nn.ModuleDict()
        for output_name, output_info in self.output_config_dict.items():
            self.output_layer_dict[output_name] = nn.Linear(self.total_subnet_output_length[output_name],
                                                            self.label_info_dict[output_name]['count'])

    def __get_subnet(self, subnet_config):
        """Get subnet according to subnet config."""
        feature = subnet_config['input']
        params = subnet_config['params']
        embedding_size = self.embedding_dict[feature]['embedding_size']
        if self.embedding_dict[feature]['type'] == 'multi_turn_text':
            self.multi_turn = True
        if self.embedding_dict[feature]['type'] == 'value':
            if subnet_config['type'] == 'dnn':
                subnet = DnnNet(embedding_size, params)
            elif subnet_config['type'] == 'dcn':
                subnet = DCN(embedding_size, params)
            else:
                raise NotImplementedError(f'The subnet {subnet_config["type"]} for value is not implemented.')
        else:
            # TODO: fix this embedding size after adding ner model (done, check this)
            if self.ner_model and subnet_config.get('ner', False):
                embedding_size += self.ner_model.model.tag_size
            if subnet_config['type'] == 'dnn':
                subnet = DnnNet(embedding_size, params)
            elif subnet_config['type'] == 'cnn':
                subnet = CnnNet(embedding_size, params)
            elif subnet_config['type'] == 'rnn':
                subnet = RnnNet(embedding_size, params)
            elif subnet_config['type'] == 'attention':
                subnet = RnnAttnNet(embedding_size, params)
            elif subnet_config['type'] == 'bert':
                subnet = BertNet(params)
            elif subnet_config['type'] == 'mask_self_attn':
                assert self.embedding_dict[feature]['type'] == 'multi_turn_text'
                subnet = MaskSelfAttn(embedding_size, params)
            else:
                raise NotImplementedError(f'The subnet {subnet_config["type"]} is not implemented.')

        return subnet

    @property
    def subnet(self):
        return self.subnet_dict

    def forward(self, inputs, sess_length):
        # Get input embedding and sequence length.
        # input_embedding: [batch_size, seq_length, embedding_size]
        # seq_length: [batch_size]
        inputs_embedding, seq_length = self.embedding_layers(inputs)
        # Get subnet outputs
        subnet_outputs = self.__get_subnet_outputs(inputs, inputs_embedding, seq_length, sess_length)
        # Get MixNet outputs
        mixnet_outputs = {}
        # Concatenate subnet outputs.
        # subnet_cat_output: [batch_size, total_subnet_output_length]
        subnet_dict = defaultdict(list)
        for subnet_name, subnet_layer in subnet_outputs.items():
            for output_name, output_info in self.output_config_dict.items():
                if (not output_info.get('input')) or (subnet_name in output_info.get('input', [])):
                    subnet_dict[output_name].append(subnet_layer)
        subnet_cat_output = {}
        for label_name, label_info in self.label_info_dict.items():
            subnet_cat_output[label_name] = torch.cat(tuple(subnet_dict[label_name]), dim=-1)
        for label_name, output_layer in self.output_layer_dict.items():
            # Map subnet outputs.
            mixnet_outputs[label_name] = output_layer(subnet_cat_output[label_name])
        return mixnet_outputs

    def __get_subnet_outputs(self, inputs, inputs_embedding, seq_length, sess_length):
        subnet_outputs = OrderedDict()
        for name, config in self.model_config_dict.items():
            feature, subnet_type, output_name = config['input'], config['type'], config['output']
            # output: [batch_size, output_size]
            # output: [sentence_count, output_size] if inputs are with multi-turn type.
            if self.embedding_dict[feature]['type'] == 'value':
                enum_emb_inputs = inputs_embedding[f'{feature}_enum']
                value_emb_inputs = inputs[feature]['value']
                subnet_emb_inputs = torch.cat((enum_emb_inputs, value_emb_inputs), dim=1)
                if subnet_type == 'dnn':
                    value_output = self.subnet_dict[name](subnet_emb_inputs)
                elif subnet_type == 'dcn':
                    value_output = self.subnet_dict[name](subnet_emb_inputs)
                else:
                    raise NotImplementedError(f'The subnet {subnet_type} of value feature is not implemented.')
                if self.multi_turn:
                    value_output = torch.repeat_interleave(value_output, sess_length, dim=0)
                subnet_outputs[output_name] = value_output
            else:
                subnet_emb_inputs = inputs_embedding[feature] if feature in inputs_embedding else None
                if config.get('ner', False) and self.ner_model:
                    ner_output = self.ner_model.model(inputs[feature])
                    subnet_emb_inputs = torch.cat((subnet_emb_inputs, ner_output), dim=-1)
                    # TODO: check here
                subnet_seq_length = seq_length[feature] if feature in seq_length else None
                if subnet_type in ['dnn', 'cnn']:
                    output = self.subnet_dict[name](subnet_emb_inputs)
                elif subnet_type in ['rnn', 'attention']:
                    output = self.subnet_dict[name](subnet_emb_inputs, subnet_seq_length)
                elif subnet_type == 'bert':
                    output = self.subnet_dict[name](inputs[feature])
                elif subnet_type == 'mask_self_attn':
                    bert_input = inputs.get(f'{feature}_msa_bert', None)
                    output = self.subnet_dict[name](subnet_emb_inputs, bert_input, subnet_seq_length, sess_length)
                else:
                    raise NotImplementedError(f'The subnet {subnet_type} is not implemented.')
                if self.multi_turn and self.embedding_dict[feature]['type'] != 'multi_turn_text':
                    output = torch.repeat_interleave(output, sess_length, dim=0)
                subnet_outputs[output_name] = output

        return subnet_outputs
