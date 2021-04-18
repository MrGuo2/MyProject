import torch
import torch.nn as nn


class CrossNet(nn.Module):
    def __init__(self, embedding_size, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        weights = []
        biases = []
        for i in range(self.num_layers):
            weights.append(nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(embedding_size, 1))))
            biases.append(nn.Parameter(torch.nn.init.constant_(torch.empty(embedding_size, 1), 0)))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, inputs):
        # inputs: [batch_size, embedding_size, 1]
        inputs = torch.unsqueeze(inputs, -1)
        # outputs: [batch_size, embedding_size, 1]
        outputs = inputs
        for i in range(self.num_layers):
            # x_w: [batch_size, 1, 1]
            x_w = torch.matmul(outputs.transpose(1, 2), self.weights[i])
            # outputs: [batch_size, embedding_size, 1]
            outputs = torch.bmm(inputs, x_w) + self.biases[i] + outputs
        # outputs: [batch_size, embedding_size]
        outputs = outputs.squeeze(-1)
        return outputs
