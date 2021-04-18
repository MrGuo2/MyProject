import copy
import os

from features.feature_provider import get_feature_handler
from utils.file import enumerate_file, enumerate_file_list, load_json, save_json
from utils.log import logger
from utils.time_track import time_desc_decorator


class FeatureIndex:
    """Feature index class."""
    def __init__(self, config, tokenizer=None):
        logger.info('Init feature index.')
        self.config = config
        self.action = config.action
        self.config_dir = config.config_dir
        self.feature_config_dict = config.feature_config_dict
        self.label_config_dict = config.label_config_dict
        self.max_index = config.max_index
        self.multi_turn = config.multi_turn
        self.tokenizer = tokenizer

        self.index_file = config.index_file
        self.index_meta_file = config.index_meta_file

        self.column_separator = config.column_separator
        # self.multi_label_separator = config.multi_label_separator
        self.multi_turn_separator = config.multi_turn_separator
        self.seq_seperator = config.seq_separator
        self.value_seperator = config.value_separator

        self.__feature_info_dict = {}
        # self.__feature_value2id = {}
        # self.__label_id2value = {}
        self.__label_info_dict = {}
        # self.__label_value2id = {}
        self.__feature_handlers = {}
        self.__label_handlers = {}
        # self.__enum_factor_keys_mapping = {}

    @property
    def feature_info_dict(self):
        return self.__feature_info_dict

    @property
    def feature_value2id(self):
        return self.__feature_value2id

    @property
    def label_handlers(self):
        return self.__label_handlers

    @property
    def feature_handlers(self):
        return self.__feature_handlers

    # @property
    # def label_id2value(self):
    #     return self.__label_id2value

    # @property
    # def label_value2id(self):
    #     return self.__label_value2id

    @property
    def label_info_dict(self):
        return self.__label_info_dict

    @time_desc_decorator('building index')
    def build_index(self, data_file_list=None):
        """Building index.

        Training mode:
            * If index file and index mata file exist, this method will load the index from the files.
            * Otherwise, this method builds the index from list of data files.

        Test mode:
            * If index file and index mata file exist, this method will load the index from the files.
            * Otherwise, raise Error.

        Args:
            data_file_list: list, list of data files.

        Returns:
            None.
        """
        if self.action == 'train':
            if os.path.exists(self.index_file) and os.path.exists(self.index_meta_file):
                self.load_index()
            else:
                if data_file_list is None:
                    raise ValueError(f'The data_file_list is None.')
                self.__build_index(data_file_list)
        elif self.action in ['predict', 'test']:
            assert os.path.exists(self.index_file) and os.path.exists(self.index_meta_file), \
                f'Index file {self.index_file} and index_meta_file {self.index_meta_file} does not exist.'
            self.load_index()

        # TODO: confirm whether __label_id2value is used in other places.
        # for name, value2id in self.__label_value2id.items():
        #     id2value = {y: x for x, y in value2id.items()}
        #     self.__label_id2value[name] = id2value
        # Building info dict.
        self.__build_info_dict()

    def load_index(self):
        """Loading index from index file and index meta file."""
        # Load index file.
        index_dict = load_json(self.index_file)
        feature_value2stat = index_dict['feature']
        label_value2stat = index_dict['label']
        # Load index meta file.
        index_meta_dict = load_json(self.index_meta_file)
        logger.info(f'Feature meta: {index_meta_dict["feature"]}')
        logger.info(f'Label meta: {index_meta_dict["label"]}')
        # Extract index.
        # self.__extract_feature_index(feature_value2stat)

        for feature_name, feature_config in self.feature_config_dict.items():
            self.__feature_handlers[feature_name] = get_feature_handler(feature_config['type'], feature_config, self.config, self.tokenizer)
            self.__feature_handlers[feature_name].load(feature_value2stat[feature_name])

        for label_name, label_config in self.label_config_dict.items():
            self.__label_handlers[label_name] = get_feature_handler(label_config['type'], label_config, self.config, self.tokenizer)
            self.__label_handlers[label_name].load(label_value2stat[label_name])

    # TODO: feature selection
    def __build_index(self, data_file_list):
        """Building index from list of data files."""
        # Init intermediate variable.
        # self.__feature_value2stat = {}
        self.__feature_handlers = {}
        self.__label_handlers = {}
        for feature_name, feature_config in self.feature_config_dict.items():
            self.__feature_handlers[feature_name] = get_feature_handler(feature_config['type'], feature_config, self.config, self.tokenizer)
            # if feature_config['type'] == 'value':
            #     self.__feature_value2stat[f'{name}_enum'] = {}
            #     self.__feature_value2stat[f'{name}_value'] = {}
            # else:
            #     self.__feature_value2stat[name] = {}
        for label_name, label_config in self.label_config_dict.items():
            self.__label_handlers[label_name] = get_feature_handler(label_config['type'], label_config, self.config, self.tokenizer)
        # Update intermediate variable.
        column_num = None
        for line_no, line in enumerate_file_list(data_file_list):
            elements = line.split(self.column_separator)
            if column_num is None:
                column_num = len(elements)
            assert column_num == len(elements), f'The length of tokens {len(elements)} is not equal with {column_num}.'
            assert len(elements) > self.max_index, \
                (f'line {line_no} Format error: the length of tokens {len(elements)}'
                 f'should be larger than {self.max_index}.')
            # Update label index.
            self.__update_label_stat(elements)
            # Update feature counts.
            self.__update_feature_stat(elements)
        # Build index.
        self.__build_index_by_stat()
        # Save index.
        self.__save_index()
        # Extract feature and label index information.
        # self.__extract_feature_index(self.__feature_value2stat)
        # Delete intermediate variable.
        # del self.__feature_value2stat
        # del self.__label_value2stat

    def __build_index_by_stat(self):
        for feature_name, handler in self.__feature_handlers.items():
            handler.build_index()

        for label_name, handler in self.__label_handlers.items():
            handler.build_index()

    def __build_info_dict(self):
        """Building info dict used for building model."""
        for feature_name, handler in self.__feature_handlers.items():
            self.__feature_info_dict[feature_name] = handler.info()

        for label_name, handler in self.__label_handlers.items():
            self.__label_info_dict[label_name] = {
                'count': len(handler)
            }

    def __save_index(self):
        """Save index to index file and index meta file, and building feature index according the order of its count."""
        # Save index file.
        save_index_dict = {
            'feature': {feature: handler.save() for feature, handler in self.__feature_handlers.items()},
            'label': {label: handler.save() for label, handler in self.__label_handlers.items()}
        }
        save_json(save_index_dict, self.index_file)

        # Save index meta file.
        save_index_meta_dict = {
            'feature': {feature_name: len(handler) for feature_name, handler in self.__feature_handlers.items()},
            'label': {label_name: len(handler) for label_name, handler in self.__label_handlers.items()}
        }
        save_json(save_index_meta_dict, self.index_meta_file)

    def __update_feature_stat(self, elements):
        """Update feature counts."""
        for feature_name, feature_config in self.feature_config_dict.items():
            idx = feature_config['index']
            value = elements[idx]
            self.__feature_handlers[feature_name].update(value)

    def __update_label_stat(self, elements):
        """Update label index."""
        for label_name, label_config in self.label_config_dict.items():
            idx = label_config['index']
            value = elements[idx]
            self.__label_handlers[label_name].update(value)

            # if self.multi_turn:
            #     label_list = elements[idx].split(self.multi_turn_separator)
            # else:
            #     label_list = [elements[idx]]
            # update label stat dict.

            # TODO: implement sim handler
            # elif label_config['type'] == 'sim':
            #     continue
            # else:
            #    raise NotImplementedError(f'The label type is not in [label, sim].')
