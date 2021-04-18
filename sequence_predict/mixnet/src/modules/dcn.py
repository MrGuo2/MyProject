import torch
import torch.nn as nn

from .cross_net import CrossNet
from .dnn import DnnNet
from utils.log import logger


class DCN(nn.Module):
    def __init__(self, embedding_size, params):
        super(DCN, self).__init__()
        self.use_dnn = params.get("use_dnn")
        self.use_cross_net = params.get("use_cross_net")
        self.embedding_size = embedding_size

        dnn_params = params.get("dnn_params", {})
        self.dnn = DnnNet(self.embedding_size, dnn_params)

        self.cn_layers = params.get("cross_net_layers", 4)
        self.cross_net = CrossNet(self.embedding_size, self.cn_layers)

        if self.use_dnn and self.use_cross_net:
            logger.info(f"Use dnn and cross_net")
            self.__output_size = self.embedding_size + self.dnn.output_size
        elif self.use_dnn:
            logger.info(f"Use dnn only")
            self.__output_size = self.dnn.output_size
        elif self.use_cross_net:
            logger.info(f"Use cross_net only")
            self.__output_size = self.embedding_size
        else:
            raise ValueError('The values of use_dnn and use_cross_net are both False.')

    @property
    def output_size(self):
        return self.__output_size

    def forward(self, inputs):
        if self.use_dnn and self.use_cross_net:
            # dnn_outputs: [batch_size, dnn_output_size]
            dnn_outputs = self.dnn(inputs)
            # cross_net_outputs: [batch_size, embedding_size]
            cross_net_outputs = self.cross_net(inputs)
            # outputs: [batch_size, dnn_output_size + embedding_size]
            outputs = torch.cat((dnn_outputs, cross_net_outputs), dim=1)
        elif self.use_dnn:
            # dnn_outputs: [batch_size, dnn_output_size]
            dnn_outputs = self.dnn(inputs)
            outputs = dnn_outputs
        elif self.use_cross_net:
            # cross_net_outputs: [batch_size, embedding_size]
            cross_net_outputs = self.cross_net(inputs)
            outputs = cross_net_outputs
        else:
            raise ValueError('The values of use_dnn and use_cross_net are both False.')

        return outputs
