import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .attention import ScaleDotAttention
from .bert import BertNet
from .cnn import CnnNet
from .dnn import DnnNet
from .rnn import RnnAttnNet, RnnNet


class MaskSelfAttn(nn.Module):
    """Mask self attention model.

    This model contains:
        * Sentence encoders: like cnn, lstm, etc.
        * Context mask self attention layer.
    """

    def __init__(self, embedding_size, params):
        super(MaskSelfAttn, self).__init__()
        self.embedding_size = embedding_size
        self.input_models = params.get('inputs')
        self.dropout_rate = params.get('dropout_rate', 0.5)
        # Create subnets.
        self.subnet_dict = nn.ModuleDict()
        self.__output_size = 0
        for subnet_name, subnet_params in self.input_models.items():
            subnet = self.__get_subnet(subnet_name, subnet_params)
            self.subnet_dict[subnet_name] = subnet
            self.__output_size += subnet.output_size

        self.attention_layer = ScaleDotAttention(self.__output_size, self.dropout_rate)

    def __get_subnet(self, subnet_name, subnet_params):
        """Get subnet according to subnet config."""
        if subnet_name == 'dnn':
            subnet = DnnNet(self.embedding_size, subnet_params)
        elif subnet_name == 'cnn':
            subnet = CnnNet(self.embedding_size, subnet_params)
        elif subnet_name == 'rnn':
            subnet = RnnNet(self.embedding_size, subnet_params)
        elif subnet_name == 'attention':
            subnet = RnnAttnNet(self.embedding_size, subnet_params)
        elif subnet_name == 'bert':
            subnet = BertNet(subnet_params)
        else:
            raise NotImplementedError(f'The subnet {subnet_name} in mask self attention is not implemented.')

        return subnet

    @property
    def output_size(self):
        return self.__output_size

    @property
    def subnet(self):
        return self.subnet_dict

    def forward(self, inputs, bert_input, seq_length, sess_length):
        # inputs: [sentence_count, embedding_size]
        # sess_length: [batch_size]
        subnet_outputs = self.__get_subnet_outputs(inputs, bert_input, seq_length)
        # Convert 2D subnet_outputs to 3D sess_inputs.
        # sess_inputs: [batch_size, max_sess_length, output_size]
        sess_inputs = pad_sequence(torch.split(subnet_outputs, sess_length.tolist()),
                                   batch_first=True,
                                   padding_value=0)
        max_sess_length = sess_inputs.size(1)
        # attn_mask: [max_sess_length, max_sess_length]
        attn_mask = inputs.new_full((max_sess_length, max_sess_length), fill_value=float('-inf')).triu(1)
        # outputs: [batch_size, max_sess_length, output_size]
        outputs = self.attention_layer(query=sess_inputs,
                                       key=sess_inputs,
                                       value=sess_inputs,
                                       attn_mask=attn_mask)
        # The inverse operation of pad_sequence.
        # output: [sentence_count, output_size]
        outputs = torch.cat(tuple([ele[:sess_length[i]] for i, ele in enumerate(outputs)]), dim=0)

        return outputs

    def __get_subnet_outputs(self, inputs, bert_input, seq_length):
        subnet_output_list = []
        for subnet_name, subnet in self.subnet_dict.items():
            if subnet_name in ['dnn', 'cnn']:
                subnet_output_list.append(subnet(inputs))
            elif subnet_name in ['rnn', 'attention']:
                subnet_output_list.append(subnet(inputs, seq_length))
            elif subnet_name == 'bert':
                subnet_output_list.append(subnet(bert_input))
            else:
                raise NotImplementedError(f'The subnet {subnet_name} in mask self attention is not implemented.')
        subnet_outputs = torch.cat(tuple(subnet_output_list), dim=-1)

        return subnet_outputs
