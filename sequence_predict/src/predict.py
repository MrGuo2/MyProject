import xgboost as xgb
import json
import pandas as pd
import numpy as np
import os

JSON_FILE = '/home/wb-ghd629671/sequence_prediction/huabei/config/predict.json'


class LoadModel(object):
    def __init__(self):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        for key, value in data_json.items():
            setattr(self, key, value)

    def load_model(self):
        # load model and data
        model_filename = os.path.join(self.data_dir_path, self.model_file)
        input_filename = os.path.join(self.data_dir_path, self.input_file)
        output_filename = os.path.join(self.data_dir_path, self.output_file)
        load_model = xgb.Booster(model_file=model_filename)
        # Data preprocess
        test_data = np.load(input_filename)["arr_0"]
        print(test_data[0])
        test_data = xgb.DMatrix(test_data)
        # Predict and output data
        predict_data = load_model.predict(test_data)
        print(predict_data[0])
        predict_data = pd.DataFrame(predict_data)
        predict_data.to_csv(output_filename, header=None, index=None)


if __name__ == '__main__':
    load_model = LoadModel()
    load_model.load_model()
