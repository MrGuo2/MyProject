import argparse
import copy
import json
import os
import shutil

from utils.file import load_json

# TODO: move these global variables in Config class or Argparse.
INDEX_FILE = 'index.json'
INDEX_META_FILE = 'index.meta.json'
JIEBA_DICT_FILE = 'userdict.dic'
DESC_JSON_PATH = 'data_desc.json'
MODEL_TAR_NAME = 'model.tar.gz'
STOPWORD_FILE = 'stopwords.txt'

FEATURE_TYPE_LIST = ['multi_turn_text', 'seq', 'text', 'value', 'bert', 'vector']
LABEL_TYPE_LIST = ['label', 'multi_label', 'sim']


class Config(object):
    """Config class."""

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        assert self.column_separator != self.multi_label_separator
        assert self.column_separator != self.seq_separator
        assert self.column_separator != self.value_separator
        self.__init_files()
        self.__read_data_json()

    def __init_files(self):
        """Init directories and files for training or evaluating."""
        os.makedirs(self.temp_dir, exist_ok=True)
        if self.action == 'train':
            assert os.path.exists(self.config_dir), f'Config directory {self.config_dir} does not exist.'
            shutil.rmtree(self.tb_log_dir, ignore_errors=True)
            os.makedirs(self.tb_log_dir, exist_ok=True)
            if self.preprocess_data:
                shutil.rmtree(self.model_dir, ignore_errors=True)
                shutil.copytree(self.config_dir, self.model_dir)
        elif self.action == 'test':
            assert os.path.exists(self.test_file), f'Test file {self.test_file} does not exist.'
            assert os.path.exists(self.model_dir)
            assert self.checkpoint is not None, f'Checkpoint file {self.checkpoint} is None.'
            self.checkpoint = os.path.join(self.model_dir, self.checkpoint)
            assert os.path.exists(self.checkpoint), f'Checkpoint file {self.checkpoint} does not exist.'
            self.config_dir = self.model_dir
        elif self.action == 'predict':
            assert os.path.exists(self.model_dir)
            assert self.checkpoint is not None, f'Checkpoint file {self.checkpoint} is None.'
            self.checkpoint = os.path.join(self.model_dir, self.checkpoint)
            assert os.path.exists(self.checkpoint), f'Checkpoint file {self.checkpoint} does not exist.'
            self.config_dir = self.model_dir

        self.index_file = os.path.join(self.model_dir, INDEX_FILE)
        self.index_meta_file = os.path.join(self.model_dir, INDEX_META_FILE)
        self.stopwords_file = os.path.join(self.model_dir, STOPWORD_FILE)
        self.jieba_dict_file = os.path.join(self.model_dir, JIEBA_DICT_FILE)

    def __read_data_json(self):
        """Read and process config json."""
        data_json_path = os.path.join(self.config_dir, DESC_JSON_PATH)
        json_config = load_json(data_json_path)
        data_config_json = copy.deepcopy(json_config['input'])
        self.model_config_dict = copy.deepcopy(json_config['net'])
        self.output_config_dict = copy.deepcopy(json_config['output'])

        # Extract feature config and label config.
        self.feature_config_dict = {}
        self.label_config_dict = {}
        self.pn_config_dict = {}

        self.remove_stopwords = False
        multi_turn_count = 0
        self.multi_turn_fea_name = ''
        for name, config in data_config_json.items():
            data_type = config.get('type', '')
            if data_type in FEATURE_TYPE_LIST:
                if data_type == 'text':
                    if 'seg_type' not in config:
                        config['seg_type'] = 'word'
                    if 'remove_stopwords' not in config:
                        config['remove_stopwords'] = False
                    elif config['remove_stopwords']:
                        self.remove_stopwords = True
                elif data_type == 'multi_turn_text':
                    multi_turn_count += 1
                    self.multi_turn_fea_name = name
                self.feature_config_dict[name] = config
            elif data_type in LABEL_TYPE_LIST:
                self.label_config_dict[name] = config
            elif data_type == 'pos':
                self.pn_config_dict[name] = config

        assert len(self.label_config_dict) > 0
        # Task count.
        self.task_num = len(self.label_config_dict)
        assert self.task_num > 0
        # Get max index.
        self.max_index = max([config['index'] for config in data_config_json.values()])
        # Set multi-turn.
        if multi_turn_count == 0:
            self.multi_turn = False
        elif multi_turn_count == 1:
            self.multi_turn = True
        else:
            raise ValueError('There are more than one multi_turn_text feature columns in the data config.')
        # Use bert.
        self.use_bert = False
        bert_feature = set()
        for name, config in self.model_config_dict.items():
            if config['type'] == 'bert':
                self.use_bert = True
                feature_name = config['input']
                if feature_name in bert_feature:
                    raise NotImplementedError(f'Multi-bert model in the same feature is not implemented.')
                else:
                    bert_feature.add(feature_name)
            if config['type'] == 'mask_self_attn' and 'bert' in config['params']['inputs']:
                self.use_bert = True

    def __str__(self):
        """Format print."""
        sorted_dict = dict(sorted(self.__dict__.items(), key=lambda v: v[0]))
        config_str = 'Configurations.\n'
        config_str += json.dumps(sorted_dict, ensure_ascii=False, indent=4)
        return config_str


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(input_args=None):
    """Get training and evaluating configs."""
    parser = argparse.ArgumentParser(description='MixNet PyTorch Parameters.')

    # Device config.
    parser.add_argument('--cpu_nums', type=int, default=10,
                        help='Maximum available CPU number (Default: %(default)s).')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Using CPU to solve the model.')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='GPU index (Default: %(default)s).')

    # Mode.
    parser.add_argument('-a', '--action', type=str, required=True, choices=['train', 'test', 'predict'],
                        help='The mode of solving model (Required).')

    # Checkpoint.
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint file with `model_{epoch}.pt` format file (Default: %(default)s).')

    # Input training file.
    parser.add_argument('-f', '--labeled_file', type=str, default='',
                        help='The input file for training and validation (Default: %(default)s).')
    parser.add_argument('-e', '--extra_train_file', type=str, default=None,
                        help='Extra training file (Default: %(default)s).')
    parser.add_argument('-v', '--valid_file', type=str, default=None,
                        help='Validation file (Default: %(default)s). '
                             'If None, the data is fetched by splitting labeled_file.')

    # Separators in the file.
    parser.add_argument('--column_separator', type=str, default='\t',
                        help='Column separator used to separate data line into some columns.')
    parser.add_argument('--multi_label_separator', type=str, default=' ',
                        help='Multi-label separator (Default: %(default)s).')
    parser.add_argument('--multi_turn_separator', type=str, default='→',
                        help='Multi-turn separator used in label, pos and multi_turn_text(Default: %(default)s).')
    parser.add_argument('--seq_separator', type=str, default=' ',
                        help='Separator for feature with `seq` type (Default: %(default)s).')
    parser.add_argument('--value_separator', type=str, default=' ',
                        help='Separator for feature with `value` type (Default: %(default)s).')

    # Directories.
    parser.add_argument('--config_dir', type=str, default='config',
                        help='Directory used to save configs (Default: %(default)s).')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory used to save model files (Default: %(default)s).')
    parser.add_argument('--tb_log_dir', type=str, default='./tensorboard',
                        help='Directory used to save TensorBoard logs (Default: %(default)s).')
    parser.add_argument('--temp_dir', type=str, default='temp',
                        help='Directory used to save temp files (Default: %(default)s).')

    # Data loading parameters.
    parser.add_argument('--ivr', action='store_true',
                        help='The input data is from IVR channel.')
    parser.add_argument('-s', '--labeled_file_split_percent', type=float, default=0.9,
                        help='Percentage of training data in labeled_file (Default: %(default)s).')
    parser.add_argument('-p', '--preprocess_data', action='store_true',
                        help='Need preprocess files, including feature index, file split, etc.')
    parser.add_argument('-r', '--read_workers', type=int, default=10,
                        help='Number of reading workers for reading batch files (Default: %(default)s).')
    parser.add_argument('--train_file_weight', type=int, default=1,
                        help='Number of train_file duplications (Default: %(default)s).')

    # Model parameters.
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--clip', type=float, default=100.,
                        help='Max norm of the gradients (Default: %(default)s).')
    parser.add_argument('-n', '--epoch_num', type=int, default=50)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-L', '--log_interval', type=int, default=1,
                        help='The interval of printing training log (Default: %(default)s).')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum used in optimizer.')
    parser.add_argument('--multi_label_threshold', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Model optimizer (Default: %(default)s).')
    parser.add_argument('--topk', type=int, default=3,
                        help='The k in "top-k" used for get top-k predictions (Default: %(default)s).')
    parser.add_argument('--valid_interval', type=int, default=1,
                        help='The interval of model validation (Default: %(default)s).')

    # Bert model parameters.
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--bert_warmup_prop', type=float, default=0.1)

    # Test.
    parser.add_argument('-t', '--test_file', type=str, default=None)
    parser.add_argument('-to', '--test_out_file', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=256)

    parser.add_argument('-RS','--ReinforcementOrSupervised', type=str,
                        default='Reinforcement', choices=['Reinforcement', 'Supervised'])

    # Parse args.
    args = parser.parse_args(input_args)

    # Namespace => Dictionary.
    kwargs = vars(args)

    return Config(**kwargs)
