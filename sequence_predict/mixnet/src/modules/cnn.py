import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnNet(nn.Module):
    """CNN model. """

    def __init__(self, embedding_size, params):
        super(CnnNet, self).__init__()
        self.dropout_rate = params.get('dropout_rate', 0.2)
        self.filter_sizes = params.get('filter_sizes', [5])
        self.input_channels = params.get('input_channels', 1)
        self.num_filters = params.get('num_filters', 32)

        self.__output_size = self.num_filters * len(self.filter_sizes)

        self.net = nn.ModuleList()
        for size in self.filter_sizes:
            conv_pool = nn.Sequential(nn.Conv2d(self.input_channels,
                                                self.num_filters,
                                                kernel_size=(size, embedding_size),
                                                padding=0),
                                      nn.ReLU())
            self.net.append(conv_pool)
        self.dropout = nn.Dropout(self.dropout_rate)

    @property
    def output_size(self):
        return self.__output_size

    def forward(self, inputs):
        max_seq_length = inputs.shape[1]
        # cnn_inputs: [batch_size, 1, max_seq_length, embedding_size]
        cnn_inputs = torch.unsqueeze(inputs, dim=1)
        cnn_outputs = []
        for size, layer in zip(self.filter_sizes, self.net):
            # output: [batch_size, num_filters, max_seq_length - size + 1, 1]
            output = layer(cnn_inputs)
            # output: [batch_size, num_filters, 1, 1]
            output = F.max_pool2d(output, kernel_size=(max_seq_length - size + 1, 1))
            cnn_outputs.append(output)
        # cnn_outputs: [batch_size, num_filters * len(filter_sizes), 1, 1]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)
        # cnn_outputs: [batch_size, num_filters * len(filter_sizes)]
        cnn_outputs = torch.reshape(cnn_outputs, (-1, self.__output_size))
        assert inputs.shape[0] == cnn_outputs.shape[0]
        cnn_outputs = self.dropout(cnn_outputs)

        return cnn_outputs
