import torch
import torch.nn as nn


class DnnNet(nn.Module):
    """DNN model."""

    def __init__(self, embedding_size, params):
        super(DnnNet, self).__init__()
        self.dropout_rate = params.get('dropout_rate', 0.3)
        self.layer_units = params.get('layers', [100, 50])
        self.neg_slope = params.get('leaky_relu_neg_slope', 0.01)

        self.__output_size = self.layer_units[-1]

        modules = []
        for i, _ in enumerate(self.layer_units):
            if i == 0:
                modules.extend([
                    nn.Linear(embedding_size, self.layer_units[i]),
                    nn.Dropout(self.dropout_rate),
                    nn.LeakyReLU(self.neg_slope)
                ])
            else:
                modules.extend([
                    nn.Linear(self.layer_units[i - 1], self.layer_units[i]),
                    nn.Dropout(self.dropout_rate),
                    nn.LeakyReLU(self.neg_slope)
                ])
        self.net = nn.Sequential(*modules)

    @property
    def output_size(self):
        return self.__output_size

    def forward(self, inputs):
        # inputs: [batch_size, embedding_size] or [batch_size, seq_length, embedding_size]
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1)
        # inputs: [batch_size, embedding_size]
        # outputs: [batch_size, output_size]
        outputs = self.net(inputs)
        return outputs
