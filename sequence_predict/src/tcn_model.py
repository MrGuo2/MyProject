# -*- coding: utf-8 -*-
import math
import keras
from keras.layers import Dense
from keras.models import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tcn import TCN
from utils.file import *
from data_preprocessing import DataPreprocess

FORMAT_PATTERN = '%Y-%m-%d %H:%M'


class TcnModel(object):

    def __init__(self, config):

        self.data_dir_path = config.data_dir_path
        for key, value in config.tcn_model.items():
            setattr(self, key, value)
        for key, value in config.data_preprocess.items():
            setattr(self, key, value)

    def run(self):

        self.tcn_model()

    def get_feature(self, input_file):

        list_x, list_y, list_x_time, list_y_time = [], [], [], []
        index = 0
        with open(input_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                data_x, data_y, time_x, time_y = line.split('#')
                list_y.append(float(data_y))
                list_y_time.append(str(time_y))
                if ',' in line:
                    list_x.append([list(map(float, arr.split(','))) for arr in data_x.split('\t')])
                    list_x_time.append(time_x.split(','))
                else:
                    list_x.append(list(map(float, data_x.split('\t'))))
                    list_x_time.append(time_x.split(','))

                if (index + 1) % 500 == 0:
                    yield list_x, list_y, list_x_time, list_y_time
                    list_x, list_y, list_x_time, list_y_time = [], [], [], []
                index += 1
            if list_x:
                yield list_x, list_y, list_x_time, list_y_time

    def get_feature2(self, input_file):
        list_y = []
        with open(input_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                data_x, data_y, time_x, time_y = line.split('#')
                list_y.append(float(data_y))
            return np.array(list_y)

    def generate_fit(self, file_path):
        while True:
            for list_x, list_y, list_time_x, list_time_y in self.get_feature(file_path):
                list_x = DataPreprocess.add_feature(file_path, list_time_x) if self.tcn_add_features else list_x
                yield np.array(list_x), np.array(list_y)

    def generate_evaluate(self, file_path):
        while True:
            for list_x, list_y, list_time_x, list_time_y in self.get_feature(file_path):
                list_x = DataPreprocess.add_feature(file_path, list_time_x) if self.tcn_add_features else list_x
                yield np.array(list_x), np.array(list_y)

    def tcn_model(self):
        input_train_file = os.path.join(self.data_dir_path, self.fit_file)
        input_test_file = os.path.join(self.data_dir_path, self.evaluate_file)
        train_step_epoch = int(math.ceil(file_len(input_train_file) / self.params["batch_size"])) - 1
        test_step_epoch = int(math.ceil(file_len(input_test_file) / self.params["batch_size"])) - 1

        input_data = Input(shape=(self.params["time_steps"], self.params["input_dim"]))
        output_data = TCN(kernel_size=self.params["kernel_size"],
                          return_sequences=self.params["return_sequences"],
                          dropout_rate=self.params["dropout_rate"])(input_data)
        output_data = Dense(1)(output_data)
        tcn_model = Model(inputs=[input_data], outputs=[output_data])
        tcn_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae')
        checkpoint = ModelCheckpoint(filepath=self.best_model_path, monitor='val_loss', save_best_only=True)
        tcn_model.fit_generator(self.generate_fit(input_train_file),
                                steps_per_epoch=train_step_epoch, epochs=self.params["epochs"], verbose=1,
                                validation_data=self.generate_evaluate(input_test_file),
                                validation_steps=test_step_epoch,
                                callbacks=[checkpoint])
        predict_fit = tcn_model.predict_generator(
            self.generate_fit(input_train_file), steps=train_step_epoch, verbose=1)
        predict_evaluate = tcn_model.predict_generator(
            self.generate_evaluate(input_test_file), steps=test_step_epoch,
            verbose=1)  # 只有值

        train_label = self.get_feature2(input_train_file)
        test_label = self.get_feature2(input_test_file)
        logger.info(f"Train model accuracy: {model_accuracy(predict_fit, train_label)}")
        logger.info(f"Test model accuracy: {model_accuracy(predict_evaluate, test_label)}")
