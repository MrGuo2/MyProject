import os
import pickle
import re
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml

from data_handler import BatchHandler
from utils.log import logger


class MixnetDataset(Dataset):
    """PyTorch MixNet dataset."""

    def __init__(self, config, data_dir, feature_index, tokenizer=None, return_line=False):
        logger.info(f'Init MixNet dataset from {data_dir}.')
        self.data_dir = data_dir
        self.batch_handler = BatchHandler(config,
                                          feature_index.feature_handlers,
                                          feature_index.label_handlers)
        self.len_dataset = len(os.listdir(self.data_dir))
        self.return_line = return_line

    def __getitem__(self, index):
        with open(os.path.join(self.data_dir, f'data_{index:06d}'), 'r', encoding='utf-8') as fin:
            lines = [line.rstrip('\n') for line in fin]
        fea2vec, label2vec, pn2vec, sess_length = self.batch_handler.handle_lines(lines)
        if self.return_line:
            return fea2vec, label2vec, pn2vec, sess_length, lines
        else:
            del lines
            return fea2vec, label2vec, pn2vec, sess_length

    def __len__(self):
        # TODO: Check this method.
        return self.len_dataset


class NERDataset(Dataset):
    """PyTorch NERdataset."""

    def __init__(self, model_path, data_path, data_type='train', tags=None, max_len=0, batch_size=128):
        self.index = 0
        self.input_size = 0
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_type = data_type
        self.vocab = {'unk': 0}
        self.data_len = 0
        self.line_offsets = []
        if not tags:
            tags = []
        self.tag_map = {'O': 0, 'START': 1, 'STOP': 2}
        if data_type == 'train':
            assert tags, Exception('请指定需要训练的tag类型，如["ORG", "PER"]')
            self.__generate_tags(tags)
        elif data_type == 'dev':
            self.load_data_map()
        elif data_type == 'test':
            self.load_data_map()
        self.data_path = data_path

        self.__prepare_data()

    def __load_data_map(self):
        with open(os.path.join(self.model_path, 'ner_data.pkl'), 'rb') as f:
            self.data_map = pickle.load(f)
            self.vocab = self.data_map.get('vocab', {})
            self.tag_map = self.data_map.get('tag_map', {})
            self.tags = self.data_map.keys()

    def __generate_tags(self, tags):
        self.tags = []
        for tag in tags:
            for prefix in ['B-', 'I-', 'E-']:
                self.tags.append(prefix + tag)
        self.tags.append('O')

    def load_data_map(self):
        with open(os.path.join(self.model_path, 'data.pkl'), 'rb') as f:
            self.data_map = pickle.load(f)
            self.vocab = self.data_map.get('vocab', {})
            self.tag_map = self.data_map.get('tag_map', {})
            self.tags = self.data_map.keys()

    def dump_data_map(self):
        with open(os.path.join(self.model_path, 'data.pkl'), 'wb') as f:
            self.data_map = {
                'vocab': self.vocab,
                'tag_map': self.tag_map,
                'line_offsets': self.line_offsets
            }
            pickle.dump(self.data_map, f)

    def __parse_line(self, line):
        ptn = re.compile(r'\((?P<start>(\d+)),(?P<end>(\d+))\)\<(?P<tag>(\w+))\|\>')
        line = line.strip()
        if not line:
            return [], [], []
        line_segs = line.split('\t')
        sentence, str_label, tags = line_segs[0], line_segs[1], line_segs[2]
        str_label = str_label.replace('没有属性', 'OTHER')
        labels = ['O'] * len(sentence)
        m = re.search(ptn, str_label)
        try:
            while m:
                start_idx, end_idx, tag = int(m.group('start')), int(m.group('end')), m.group('tag')
                if tag != 'OTHER':
                    for i in range(start_idx, end_idx + 1):
                        labels[i - 1] = ('B-' if i == start_idx else 'E-' if i == end_idx else 'I-') + tag[:3].upper()
                str_label = str_label[m.end():]
                m = re.search(ptn, str_label)
        except IndexError:
            logger.warning(f'Error parsing label:{line}')
        else:
            return list(sentence), labels, tags.split(',')
        return [], [], []

    def __prepare_data(self):
        self.data_len = 0
        logger.info('Preparing data...')
        max_len = 0
        offset = 0
        with open(self.data_path) as f:
            for line in tqdm(f):
                sentence, labels, tags = self.__parse_line(line)
                if not sentence:
                    offset += len(line.encode())
                    continue
                if not self.max_len:
                    max_len = max(max_len, len(sentence))
                for (word_id, word) in enumerate(sentence):
                    tag = labels[word_id]
                    if word not in self.vocab and self.data_type == 'train':
                        self.vocab[word] = max(self.vocab.values()) + 1
                    if tag not in self.tag_map and self.data_type == 'train' and tag in self.tags:
                        self.tag_map[tag] = len(self.tag_map.keys())
                self.data_len += 1
                self.line_offsets.append(offset)
                offset += len(line.encode())
        self.input_size = len(self.vocab.values())
        if not self.max_len:
            self.max_len = max_len
        logger.info(f'{self.data_type} data: {self.data_len}')
        logger.info(f'vocab size: {self.input_size}')
        logger.info(f'sentence max length: {self.max_len}')
        logger.info(f'unique tag: {len(self.tag_map.values())}')
        logger.info(f'tags: {self.tag_map}')
        logger.info('=' * 64)

    def __pad_data(self, data_row):
        # pad or truncate
        if len(data_row[0]) < self.max_len:
            data_row[0] = data_row[0] + (self.max_len - len(data_row[0])) * [0]
            data_row[1] = data_row[1] + (self.max_len - len(data_row[1])) * [0]
        elif len(data_row[0]) > self.max_len:
            data_row[0] = data_row[0][:self.max_len]
            data_row[1] = data_row[1][:self.max_len]
        return data_row

    def __getitem__(self, index):
        with open(self.data_path) as f:
            f.seek(self.line_offsets[index])
            line = f.readline()
            sentence, labels, tags = self.__parse_line(line)
            if not sentence:
                raise RuntimeError(f'Data line#{index} offset error:{self.line_offsets[index]}')
            data_row = [
                list(map(lambda x: self.vocab.get(x), sentence)),
                list(map(lambda x: self.tag_map.get(x), labels))
            ]
            if None in data_row[0]:
                raise RuntimeError('Can\'t find some sentence words in vocab.')
            if None in data_row[1]:
                raise RuntimeError('Can\'t find some tags in tag_map.')
            data_row = self.__pad_data(data_row)
        return data_row

    def __len__(self):
        return self.data_len

    def batch_iter(self):
        batch = []
        with open(self.data_path) as f:
            for index in range(self.data_len):
                f.seek(self.line_offsets[index])
                line = f.readline()
                sentence, labels, tags = self.__parse_line(line)
                if not sentence:
                    raise RuntimeError(f'Data line#{index} offset error:{self.line_offsets[index]}')
                data_row = [
                    list(map(lambda x: self.vocab.get(x), sentence)),
                    list(map(lambda x: self.tag_map.get(x), labels))
                ]
                if None in data_row[0]:
                    raise RuntimeError('Can\'t find some sentence words in vocab.')
                if None in data_row[1]:
                    raise RuntimeError('Can\'t find some tags in tag_map.')
                data_row = self.__pad_data(data_row)
                batch.append(data_row)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


if __name__ == '__main__':
    config = yaml.load(open('config/ner_config.yml', 'r'))
    ner_dataset = NERDataset(config.get('model_path'),
                             'data/ner_train.txt',
                             'train',
                             config.get('tags'),
                             max_len=200,
                             batch_size=config.get('batch_size'))
    print(config)
    for data in tqdm(ner_dataset):
        pass
