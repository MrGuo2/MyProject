import codecs
import copy
import os
import pickle
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
import yaml

from dataset import NERDataset
from utils.log import logger

START_TAG = 'START'
STOP_TAG = 'STOP'


def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTMCRF(nn.Module):

    def __init__(
            self,
            device,
            tag_map=None,
            batch_size=128,
            vocab_size=200,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100
    ):
        super(BiLSTMCRF, self).__init__()
        if not tag_map:
            tag_map = {'O': 0, 'B-COM': 1, 'I-COM': 2, 'E-COM': 3, 'START': 4, 'STOP': 5}
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.tag_size = len(tag_map)
        self.tag_map = tag_map

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)
        # self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2).to(self.device))

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)
        return logits

    def forward(self, sentences, lengths=None):
        """
        :params sentences sentences to predict
        :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        sentences = torch.tensor(sentences, dtype=torch.long).to(self.device)
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        return logits


class NERModel(object):

    def __init__(self, device, entry='train'):
        self.device = device
        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == 'train':
            self.train_manager = NERDataset(model_path=self.model_path,
                                            data_path='data/ner_train.txt',
                                            data_type='train',
                                            tags=self.tags,
                                            max_len=self.embedding_size,
                                            batch_size=self.batch_size)
            self.train_manager.dump_data_map()
            self.total_size = (len(self.train_manager) + self.batch_size - 1) // self.batch_size
            dev_manager = NERDataset(model_path=self.model_path,
                                     data_path='data/ner_test.txt',
                                     data_type='dev',
                                     tags=self.tags,
                                     max_len=self.embedding_size,
                                     batch_size=self.batch_size)
            self.dev_batch = dev_manager.batch_iter()

            self.model = BiLSTMCRF(
                self.device,
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()
        elif entry == 'predict':
            data_map = self.load_params()
            self.tag_map = data_map.get('tag_map')
            self.vocab = data_map.get('vocab')
            self.model = BiLSTMCRF(
                self.device,
                tag_map=self.tag_map,
                vocab_size=len(self.vocab.items()),
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()
        self.model.to(self.device)

    def load_config(self):
        try:
            fopen = open('config/ner_config.yml')
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            logger.warning(f'Load config failed, using default config {error}')
            with open('config/ner_config.yml', 'w') as fopen:
                config = {
                    'embedding_size': 200,
                    'hidden_size': 128,
                    'batch_size': 128,
                    'dropout': 0.5,
                    'model_path': 'model/',
                    'tags': ['ORG', 'PER', 'LOC', 'COM']
                }
                yaml.dump(config, fopen)
        self.embedding_size = config.get('embedding_size')
        self.hidden_size = config.get('hidden_size')
        self.batch_size = config.get('batch_size')
        self.model_path = config.get('model_path')
        self.tags = config.get('tags')
        self.dropout = config.get('dropout')

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'params.pkl')))
            logger.info('model restore success!')
        except Exception as error:
            logger.warn(f'model restore faild! {error}')

    def load_params(self):
        with codecs.open('ner_model/data.pkl', 'rb') as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        epoch_num = 1
        for epoch in range(epoch_num):
            progress = tqdm(self.train_manager.batch_iter(), desc=f'NER Epoch#{epoch + 1}/{epoch_num}',
                            total=self.total_size, dynamic_ncols=True)
            for batch in progress:
                self.model.zero_grad()
                sentences, tags = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(self.device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(self.device)
                trained_tags = self.model(sentences_tensor)
                loss = -self.model.crf(trained_tags, tags_tensor)  # neg_log_likelihood
                progress.set_postfix({
                    'loss': loss.item(),
                })
                loss.backward()
                optimizer.step()
            torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

    def evaluate(self):
        sentences, labels = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        for tag in self.tags:
            pass
            # f1_score(labels, paths, tag, self.model.tag_map)

    def predict(self, input_str=''):
        if not input_str:
            input_str = input('请输入文本: ')
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).to(self.device).view(1, -1)
        # _, paths = self.model(sentences)
        id2tag = [k for (k, v) in sorted(self.tag_map.items(), key=lambda x: x[1])]
        results = {}
        for tag in id2tag:
            results.update({
                tag.split('-')[-1]: []
            })
        trained_tags = self.model(sentences)
        entities = self.model.crf.decode(trained_tags)
        tags = list(map(lambda x: id2tag[x[0]], entities))
        return tags


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('menu:\n\ttrain\n\tpredict')
        exit()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Traning NER via device:{device}')
    if sys.argv[1] == 'train':
        ner = NERModel(device, 'train')
        ner.train()
    elif sys.argv[1] == 'predict':
        ner = NERModel(device, 'predict')
        predict_list = [
            '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚',
            '喂你好哎我想问一下就是我昨天那个手机丢了我现在登我的支付宝呃他那台手机就是我我已经信息到支付宝那那另外一台手机还能不能呃登录进去的我怕他盗用我的基金',
            '为什么在中小学平台查询不到需缴费账单',
            '蚂蚁森林怎样屏蔽好友',
            '兑换蚂蚁宝卡流量可以换个手机号嘛',
            '蚂蚁借呗可以先还利息再还本金吗',
            '为什么操作预约兑换时需要确认“个人购汇申请书”']
        for text in predict_list:
            print(text)
            print(ner.predict(text))
