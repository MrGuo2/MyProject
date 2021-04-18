import os
import shutil

from utils.command import run_command
from utils.file import combine_file, file_len, split_data_percent, shuffle_by_command
from utils.log import logger
from utils.time_track import time_desc_decorator


class PreProcess(object):
    """Class for pre-processing data files."""
    def __init__(self, config):
        logger.info('Init preprocess data.')
        self.batch_size = config.batch_size
        self.preprocess_data = config.preprocess_data
        self.temp_dir = config.temp_dir
        if config.action == 'train':
            self.extra_train_file = config.extra_train_file
            self.labeled_file = config.labeled_file
            self.valid_file = config.valid_file
            self.train_percent = config.labeled_file_split_percent
            self.train_weight = config.train_file_weight
        elif config.action == 'test':
            self.test_file = config.test_file
            self.test_batch_size = config.test_batch_size

    @time_desc_decorator('preprocess training data')
    def train_preprocess(self):
        """Preprocess training and validation data."""
        train_data_dir = os.path.join(self.temp_dir, 'train_data')
        valid_data_dir = os.path.join(self.temp_dir, 'valid_data')
        # Prepare training data and validation data.
        final_train_file, final_valid_file = self.__prepare_training_file()
        # Get batch size in existing file.
        file_batch_size = 0
        if os.path.exists(f'{train_data_dir}/data_000000'):
            file_batch_size = file_len(f'{train_data_dir}/data_000000')
        logger.info(f'The batch size in existing file is {file_batch_size}.')
        # Split each batch data to files.
        if self.preprocess_data or file_batch_size != self.batch_size:
            shutil.rmtree(train_data_dir, ignore_errors=True)
            shutil.rmtree(valid_data_dir, ignore_errors=True)
            os.makedirs(train_data_dir, exist_ok=True)
            os.makedirs(valid_data_dir, exist_ok=True)
            run_command(f'split -dl{self.batch_size} -a6 {final_train_file} {train_data_dir}/data_')
            run_command(f'split -dl{self.batch_size} -a6 {final_valid_file} {valid_data_dir}/data_')

        return train_data_dir, valid_data_dir, final_train_file, final_valid_file

    @time_desc_decorator('preprocess test data')
    def test_preprocess(self):
        """Preprocess test data."""
        test_data_dir = os.path.join(self.temp_dir, 'test_data')
        # Get batch size in existing file.
        file_batch_size = 0
        if os.path.exists(f'{test_data_dir}/data_000000'):
            file_batch_size = file_len(f'{test_data_dir}/data_000000')
        logger.info(f'The batch size in existing file is {file_batch_size}.')
        # Split each batch data to files.
        if self.preprocess_data or file_batch_size != self.test_batch_size:
            shutil.rmtree(test_data_dir, ignore_errors=True)
            os.makedirs(test_data_dir, exist_ok=True)
            run_command(f'split -dl{self.test_batch_size} -a6 {self.test_file} {test_data_dir}/data_')

        return test_data_dir

    def __prepare_training_file(self):
        """Prepare training file.

            * Split labeled file to labeled train file and labeled valid file if necessary.
            * Combine labeled train file and extra train file if necessary.
            * Shuffle train file if necessary.

        Returns:
            final_train_file: Final training file path.
            final_valid_file: Final validation file path.
        """
        logger.info('Starting to prepare training data.')

        if self.valid_file is None:
            labeled_file_name = os.path.basename(self.labeled_file)
            labeled_train_file = os.path.join(self.temp_dir, f'{labeled_file_name}.train')
            labeled_valid_file = os.path.join(self.temp_dir, f'{labeled_file_name}.val')
            if self.preprocess_data or not (os.path.exists(labeled_train_file) and os.path.exists(labeled_valid_file)):
                split_data_percent(self.labeled_file, labeled_train_file, labeled_valid_file, self.train_percent)
        else:
            labeled_train_file = self.labeled_file
            labeled_valid_file = self.valid_file

        if self.extra_train_file is None:
            final_train_file = labeled_train_file
        else:
            final_train_file = os.path.join(self.temp_dir, f'{os.path.basename(labeled_train_file)}.comb')
            if self.preprocess_data or not os.path.exists(final_train_file):
                combine_file(labeled_train_file, self.extra_train_file, final_train_file, self.train_weight)
        # Shuffle training file.
        shuffle_file = os.path.join(self.temp_dir, f'{os.path.basename(final_train_file)}.shuf')
        if self.preprocess_data or not os.path.exists(shuffle_file):
            shuffle_by_command(final_train_file, shuffle_file, self.temp_dir)

        final_train_file = shuffle_file
        final_valid_file = labeled_valid_file

        return final_train_file, final_valid_file
