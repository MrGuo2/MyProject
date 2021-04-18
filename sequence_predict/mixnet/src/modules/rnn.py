import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RnnNet(nn.Module):
    """RNN model, containing vanilla RNN, LSTM, GRU. """

    def __init__(self, embedding_size, params):
        super(RnnNet, self).__init__()
        self.bidirectional = params.get('bidirectional', False)
        self.dropout_rate = params.get('dropout_rate', 0.4)
        self.hidden_size = params.get('hidden_size', 50)
        self.num_layers = params.get('num_layers', 1)
        self.pooling_type = params.get('pooling_type', 'last')
        self.rnn_type = params.get('rnn_type', 'lstm').lower()

        self.num_directions = 2 if self.bidirectional else 1
        self.__output_size = self.hidden_size * self.num_directions

        rnn_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        if self.rnn_type in rnn_dict:
            rnn_model = rnn_dict[self.rnn_type]
            self.net = rnn_model(embedding_size,
                                 self.hidden_size,
                                 self.num_layers,
                                 batch_first=False,
                                 dropout=self.dropout_rate,
                                 bidirectional=self.bidirectional)
        else:
            raise NotImplementedError(f'The rnn_type {self.rnn_type} is not implemented.')
        self.dropout = nn.Dropout(self.dropout_rate)
        self._reset_parameters()

    def _reset_parameters(self):
        """Reset RNN weight and bias parameters."""
        for name, param in self.net.named_parameters():
            # orthogonal initialization for hidden weights
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # zero initialization for hidden bias.
            if 'bias_hh' in name:
                param.data.fill_(0.0)

    @property
    def output_size(self):
        return self.__output_size

    def forward(self, inputs, seq_length):
        batch_size, max_seq_length, embedding_size = inputs.size()
        # outputs: [max_seq_length, batch_size, num_directions * hidden_size]
        outputs = self._get_rnn_outputs(inputs, seq_length)
        # last_outputs: [batch_size, num_directions * hidden_size]
        if self.pooling_type == 'last':
            # last_indices: [1, batch_size, num_directions * hidden_size]
            last_indices = (seq_length - 1).view(-1, 1).expand(batch_size, self.__output_size).unsqueeze(0)
            last_outputs = outputs.gather(0, last_indices).squeeze(0)
        elif self.pooling_type == 'mean':
            last_outputs = outputs.sum(dim=0).div(seq_length.view(-1, 1).float())
        elif self.pooling_type == 'max':
            last_outputs, _ = torch.max(outputs, dim=0)
        else:
            raise NotImplementedError(f'The pooling_type {self.pooling_type} is not implemented.')
        last_outputs = self.dropout(last_outputs)

        return last_outputs

    def _get_rnn_outputs(self, inputs, seq_length):
        # inputs: [max_seq_length, batch_size, embedding_size]
        inputs = inputs.permute(1, 0, 2)
        # Sort in decreasing order of length for pack_padded_sequence().
        input_length_sorted, inputs_sorted_indices = seq_length.sort(descending=True)
        # inputs_sorted: [max_seq_length, batch_size, embedding_size]
        inputs_sorted = inputs.index_select(1, inputs_sorted_indices)

        rnn_inputs = pack_padded_sequence(inputs_sorted,
                                          input_length_sorted,
                                          batch_first=False,
                                          enforce_sorted=True)

        self.net.flatten_parameters()
        # outputs: [seq_length, batch_size, num_directions * hidden_size]
        outputs, _ = self.net(rnn_inputs)

        # outputs: [max_seq_length, batch_size, num_directions * hidden_size]
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=False)
        # Reorder outputs.
        _, inverse_indices = inputs_sorted_indices.sort(descending=False)
        # outputs: [max_seq_length, batch_size, num_directions * hidden_size]
        outputs = outputs.index_select(1, inverse_indices)

        return outputs


class RnnAttnNet(RnnNet):
    """Attention model based on RNN. """

    def __init__(self, embedding_size, params):
        super(RnnAttnNet, self).__init__(embedding_size, params)
        self.u_dim = params.get('u_dim', 30)

        self.__output_size = self.u_dim

        self.rnn_output_mapper = nn.Sequential(nn.Linear(self.hidden_size * self.num_directions, self.u_dim),
                                               nn.Dropout(self.dropout_rate),
                                               nn.Tanh())
        self.attention_weight_layer = nn.Linear(self.u_dim, 1, bias=False)

    @property
    def output_size(self):
        return self.__output_size

    def forward(self, inputs, seq_length):
        # rnn_outputs: [max_seq_length, batch_size, num_directions * hidden_size]
        rnn_outputs = super(RnnAttnNet, self)._get_rnn_outputs(inputs, seq_length)
        # rnn_outputs: [batch_size, max_seq_length, num_directions * hidden_size]
        rnn_outputs = rnn_outputs.permute(1, 0, 2)
        # rnn_outputs: [batch_size, max_seq_length, u_dim]
        rnn_outputs = self.rnn_output_mapper(rnn_outputs)
        # attn_weight: [batch_size, max_seq_length]
        attn_weight = self.attention_weight_layer(rnn_outputs).squeeze(-1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        # attn_output: [batch_size, u_dim]
        attn_output = torch.bmm(attn_weight.unsqueeze(1), rnn_outputs).squeeze(1)
        attn_output = self.dropout(attn_output)

        return attn_output
