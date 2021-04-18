import torch
from torch.utils.data import DataLoader

from configs import get_config
from dataset import MixnetDataset
from feature_index import FeatureIndex
from models import MixNet
from modules import NERModel
from preprocess import PreProcess
from solver import Solver
from utils.log import logger
from utils.tokenizer import Tokenizer


def get_device(use_cpu, gpu_id):
    """Get torch device.

    Args:
        use_gpu: bool, whether using GPU or not.
        gpu_id: int, GPU id.

    Returns:
        device: PyTorch device.
    """
    if use_cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f'The current device or PyTorch version does not support for GPU.')
        gpu_count = torch.cuda.device_count()
        if gpu_id >= 0 and gpu_id < gpu_count:
            device = torch.device('cuda', gpu_id)
        else:
            raise RuntimeError(f'The GPU {gpu_id} is invalid.')

    return device


def train(config, device, RS='Supervised'):
    # Init tokenizer.
    tokenizer = Tokenizer(config.temp_dir,
                          config.jieba_dict_file,
                          config.remove_stopwords,
                          config.stopwords_file,
                          config.ivr)
    # Init feature index.
    feature_index = FeatureIndex(config, tokenizer=tokenizer)
    file_list = [config.labeled_file]
    if config.extra_train_file is not None:
        file_list.append(config.extra_train_file)
    if config.valid_file is not None:
        file_list.append(config.valid_file)
    feature_index.build_index(file_list)
    # Preprocess data.
    pre_process = PreProcess(config)
    train_data_dir, valid_data_dir, final_train_file, final_valid_file = pre_process.train_preprocess()
    # Get PyTorch dataset.
    train_dataset = MixnetDataset(config, train_data_dir, feature_index, tokenizer)
    valid_dataset = MixnetDataset(config, valid_data_dir, feature_index, tokenizer, True)
    # Get NER model if necessary and compatible.
    need_ner = False
    for (feature, feature_config) in config.feature_config_dict.items():
        need_ner = need_ner or ("text" in feature_config.get("type", "") and
                                    feature_config.get("seg_type", "word") == "char" and
                                    feature_config.get("ner", False))
    if need_ner:
        logger.info("Enable NER, loading NER model...")
        # Use predict mode since we cannot train it without tag information.
        ner_model = NERModel(device, "predict")
    else:
        logger.info("Disable NER.")
        ner_model = None
    # Get PyTorch data loader.
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=config.read_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=config.read_workers)
    # Init model.
    model = MixNet(config.model_config_dict,
                   config.output_config_dict,
                   feature_index.feature_info_dict,
                   feature_index.label_info_dict,
                   ner_model=ner_model)
    # Train model.
    solver = Solver(config, train_data_loader, valid_data_loader, feature_index, model, device, RS)
    solver.build()
    solver.train()


def test(config, device, RS='Supervised'):
    # Init tokenizer.
    tokenizer = Tokenizer(config.temp_dir,
                          config.jieba_dict_file,
                          config.remove_stopwords,
                          config.stopwords_file,
                          config.ivr)
    # Init feature index.
    feature_index = FeatureIndex(config, tokenizer=tokenizer)
    feature_index.build_index()
    # Preprocess data.
    pre_process = PreProcess(config)
    test_data_dir = pre_process.test_preprocess()
    # Get PyTorch dataset.
    test_dataset = MixnetDataset(config, test_data_dir, feature_index, tokenizer, True)
    # Get NER model if necessary and compatible.
    need_ner = False
    for (feature, feature_config) in config.feature_config_dict.items():
        need_ner = need_ner or ("text" in feature_config.get("type", "") and
                                feature_config.get("seg_type", "word") == "char" and
                                feature_config.get("ner", False))
    if need_ner:
        logger.info("Enable NER, loading NER model...")
        # Use predict mode since we cannot train it without tag information.
        ner_model = NERModel(device, "predict")
    else:
        logger.info("Disable NER.")
        ner_model = None
    # Get PyTorch data loader.
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.read_workers)
    # Init model.
    model = MixNet(config.model_config_dict,
                   config.output_config_dict,
                   feature_index.feature_info_dict,
                   feature_index.label_info_dict,
                   ner_model=ner_model)
    # Evaluate model.
    solver = Solver(config, None, test_data_loader, feature_index, model, device, RS)
    solver.build()
    solver.evaluate()


def main():
    # Init config.
    config = get_config()
    logger.info(config)
    # Get device.
    device = get_device(config.use_cpu, gpu_id=config.gpu_id)
    # Set maximum number of threads.
    torch.set_num_threads(config.cpu_nums)
    # Train or test.
    if config.action == 'train':
        train(config, device, config.ReinforcementOrSupervised)
    elif config.action == 'test':
        test(config, device, config.ReinforcementOrSupervised)


if __name__ == '__main__':
    main()
