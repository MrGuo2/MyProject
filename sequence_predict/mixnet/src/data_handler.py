import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import torch

from utils.log import logger


class BatchHandler(object):
    def __init__(self, config, feature_handlers, label_handlers):
        logger.info('Init batch handler.')
        self.feature_config_dict = config.feature_config_dict
        self.label_config_dict = config.label_config_dict
        self.multi_turn = config.multi_turn
        self.pn_config_dict = config.pn_config_dict

        self.column_separator = config.column_separator
        self.multi_turn_separator = config.multi_turn_separator

        self.feature_handlers = feature_handlers
        self.label_handlers = label_handlers

        # self.tokenizer = tokenizer

        # TODO: mask_self_attn and bert fix
        """
        self.use_bert = config.use_bert
        if self.use_bert:
            self.bert_tokenizer = {}
            for model_name, model_config in config.model_config_dict.items():
                if model_config['type'] == 'bert':
                    feature_name = model_config['input']
                    bert_params = model_config['params']
                    bert_tokenizer = BertTokenizer.from_pretrained(bert_params['model_dir'],
                                                                   cache_dir=bert_params['cache_dir'],
                                                                   do_lower_case=bert_params['do_lower_case'])
                    self.bert_tokenizer[feature_name] = bert_tokenizer
                
                if model_config['type'] == 'mask_self_attn' and 'bert' in model_config['params']['inputs']:
                    feature_name = f'{model_config["input"]}_msa'
                    bert_params = model_config['params']['inputs']['bert']
                    bert_tokenizer = BertTokenizer.from_pretrained(bert_params['model_dir'],
                                                                   cache_dir=bert_params['cache_dir'],
                                                                   do_lower_case=bert_params['do_lower_case'])
                    self.bert_tokenizer[feature_name] = bert_tokenizer
        """
    def handle_lines(self, line_list):
        """Convert data lines into model input.

        Args:
            line_list: list, data lines.

        Returns:
            fea_name2vec: dict, containing feature Tensors.
            label2vec: dict, containing label Tensors.
            pn2vec: dict, containing Tensors of positive and negative tags.
            sess_length: list, containing session length of each line.
        """
        batch_size = len(line_list)
        arr_list = [line.split(self.column_separator) for line in line_list]

        feature_dict = {}
        label_dict = {}
        pn_dict = {}
        # Extract features in the data lines.
        for name, column_config in self.feature_config_dict.items():
            feature_dict[name] = [arr[column_config['index']] for arr in arr_list]
        # Extract labels in the data lines.
        for name, label_config in self.label_config_dict.items():
            label_dict[name] = [arr[label_config['index']] for arr in arr_list]
        for name, pn_config in self.pn_config_dict.items():
            pn_dict[name] = [arr[pn_config['index']] for arr in arr_list]
        # Get session/conversation length.
        sess_length = self.__get_sess_length(list(label_dict.values())[0], batch_size)
        # Convert features into dict of Tensor.
        fea_name2vec = self.handle_features(feature_dict, batch_size)
        # Convert labels into dict of Tensor.
        label2vec = self.handle_labels(label_dict)
        # Get sentence count.
        sentence_count = torch.sum(sess_length).item()
        # Extract positive and negative tags and convert them to dict of Tensor.
        pn2vec = self.handle_pn(pn_dict, sentence_count)

        return fea_name2vec, label2vec, pn2vec, sess_length

    def handle_features(self, feature_dict, batch_size=1):
        """Indexing features and convert them into dict of Tensor.

        Args:
            feature_dict: dict, containing feature list with same length.
            batch_size: int, batch size. Default: 1.

        Returns:
            fea_name2vec: dict, containing feature Tensors.
        """
        fea_name2vec = {}
        for feature_name, feature_config in self.feature_config_dict.items():
            feature_list = feature_dict.get(feature_name, [''] * batch_size)
            fea_name2vec[feature_name] = self.feature_handlers[feature_name].handle(feature_list)
        return fea_name2vec

    def handle_labels(self, label_dict):
        """Indexing labels and convert them into dict of Tensor.

        Args:
            label_dict: dict, containing label list with same length.
            batch_size: int, batch size.

        Returns:
            label2vec: dict, containing label Tensors.
        """
        label2vec = {}
        for label_name, column_config in self.label_config_dict.items():
            label_vec = self.label_handlers[label_name].handle(label_dict[label_name])

            # TODO: fix sim, seem there is a bug below
            # elif label_type == 'sim':
            #     label_vec = [[float(l)] for l in label_dict[label_name]]
            # else:
            #     raise NotImplementedError(f'Unknown feature type: {label_type}.')
            label2vec[label_name] = torch.tensor(label_vec)
        return label2vec

    def handle_pn(self, pn_dict, sentence_count):
        """Extract positive and negative tags and convert them to dict of Tensor.

        Args:
            pn_dict: dict, containing positive and negative tags.
            sentence_count: int, total sentence count in the batch data.

        Returns:
            pn2vec: dict, containing Tensors of positive and negative tags.
        """
        pn2vec = {}
        # Init pn2vec.
        for label_name, label_config in self.label_config_dict.items():
            if label_config['type'] != 'multi_label':
                pn2vec[label_name] = torch.ones(sentence_count)
        # update pn2vec.
        for pn_name, pn_config in self.pn_config_dict.items():
            label_name = pn_config['label']
            if label_name not in pn2vec:
                logger.warn(f'The positive and negative symbol column for {label_name} is not used.')
                continue
            # Get pn list.
            if self.multi_turn:
                pn_vec = []
                for pns in pn_dict[pn_name]:
                    for pn in pns.split(self.multi_turn_separator):
                        pn_vec.append(int(pn))
            else:
                pn_vec = [int(pns) for pns in pn_dict[pn_name]]
            pn2vec[label_name] = torch.tensor(pn_vec)
        return pn2vec

    def __get_sess_length(self, column_data, batch_size):
        if self.multi_turn:
            sess_length = []
            for data_line in column_data:
                sess_length.append(len(data_line.split(self.multi_turn_separator)))
            sess_length = torch.tensor(sess_length, dtype=torch.long)
        else:
            sess_length = torch.ones(batch_size, dtype=torch.long)
        return sess_length
