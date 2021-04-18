import argparse
import json
import os
from utils.file import load_json

CONFIG_DIR_PATH = './config'

granu_to_ties = {
    1: (970, 1440),
    5: (194, 288),
    30: (33, 48),
    60: (17, 24)
}

bus_dict = {
    'huabei': '花呗线',
    'jiebei': '借呗线',
    'licai': '理财线'
}


class Config(object):
    """Config class."""

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        config_json = load_json(f'{CONFIG_DIR_PATH}/{self.config_file}')
        config_json['data_preprocess']['granularity'] = self.granularity
        config_json['data_preprocess']['predict_size'] = self.predict_size
        config_json['data_preprocess']['business'] = self.business
        for key, value in config_json.items():
            setattr(self, key, value)
        self.build_file_path()

    def build_file_path(self):

        # Part of Data preprocess
        if self.data_preprocess['working_hours']:
            ties = granu_to_ties[self.granularity][0]
            self.data_preprocess["one_day"] = 965 + self.predict_size
        else:
            ties = granu_to_ties[self.granularity][1]
            self.data_preprocess["one_day"] = 1440

        file_prefix = f'{self.business}_{self.call_inflow}_{self.predict_size}mins_{ties}Ties_'
        file_prefix_1 = f'{self.business}_{self.call_inflow}_'
        self.data_preprocess["business_line"] = bus_dict[self.business]
        self.input_file = os.path.join(self.data_dir_path, f'{self.business}_{self.call_inflow}.csv')
        self.data_preprocess["merge_file_name"] = f"{self.business}_{self.call_inflow}.csv"
        self.data_preprocess["del_other_file"] = file_prefix_1 + "del_other_feature.csv"
        self.data_preprocess["per_min_fit_file"] = file_prefix_1 + "per_min_fit.csv"
        self.data_preprocess["per_min_evaluate_file"] = file_prefix_1 + "per_min_evaluate.csv"
        self.data_preprocess["events_file"] = f"{self.business}_event_{self.event}.csv"
        self.data_preprocess["fit_events"] = file_prefix_1 + "fit_events.csv"
        self.data_preprocess["evaluate_events"] = file_prefix_1 + "evaluate_events.csv"
        self.data_preprocess["fit_slide_window"] = file_prefix_1 + f"{self.granularity}mins_fit_slide_window.csv"
        self.data_preprocess["evaluate_slide_window"] = file_prefix_1 + f"{self.granularity}mins_eval_slide_window.csv"
        self.data_preprocess["fit_file"] = file_prefix + f"fit_csv_file.csv"
        self.data_preprocess["evaluate_file"] = file_prefix + f"evaluate_csv_file.csv"
        self.data_preprocess["fit_npz_file"] = file_prefix + "fit.npz"
        self.data_preprocess["evaluate_npz_file"] = file_prefix + "evaluate.npz"

        # Part of Da_rnn model
        self.da_rnn_model["params"].setdefault("time_step", self.data_preprocess["one_day"])
        self.da_rnn_model["dd_url"] = "https://oapi.dingtalk.com/robot/send?access_token=" \
                                      "465d14d7344a8dad4416146c53006a172f92d36979c3323eaee52b67a5f33a90"
        self.da_rnn_model["decoder_model"] = \
            os.path.join(self.data_dir_path, file_prefix + f"decoder.tp")
        self.da_rnn_model["encoder_model"] = \
            os.path.join(self.data_dir_path, file_prefix + f"encoder.tp")

        # Part of Xgboost
        core = 'GPU' if self.tree_method == "gpu_hist" else 'CPU'
        self.xgboost_model['feature_importance'] = os.path.join(self.temp_dir_path,
                                                                f'{file_prefix_1}_feature_score_{core}.csv')
        self.xgboost_model['model_dir_path'] = os.path.join(self.model_dir_path, f'{file_prefix_1}_{core}.model')
        self.xgboost_model['plot_path_name'] = os.path.join(self.image_dir_path, f'{file_prefix_1}_plot_{core}.png')
        self.xgboost_model['print_path_name'] = os.path.join(self.temp_dir_path, f'{file_prefix_1}_diff_{core}.csv')
        self.xgboost_model['everyday_accuracy_path'] = os.path.join(self.temp_dir_path,
                                                                    f'{file_prefix_1}_everyday_accuracy_{core}.csv')
        if self.tree_method == "gpu_hist":
            self.xgboost_model["const_params"]["tree_method"] = [self.tree_method]
            self.xgboost_model["const_params"]["gpu_id"] = [self.gpu_id]
        else:
            self.xgboost_model["const_params"]["tree_method"] = [self.tree_method]

        # Part of Tcn
        self.tcn_model["params"]["input_dim"] = 38 if self.business == "huabei" else 7
        self.tcn_model["best_model_path"] = os.path.join(self.best_model_dir_path, file_prefix + "_model.h5")
        self.tcn_model["params"]["time_steps"] = ties

    def __str__(self):
        """Format print."""
        sorted_dict = dict(sorted(self.__dict__.items(), key=lambda v: v[0]))
        config_str = 'Configurations.\n'
        config_str += json.dumps(sorted_dict, ensure_ascii=False, indent=4)
        return config_str


def get_configs():
    parser = argparse.ArgumentParser(description='Data Parameters.')

    # Device config.
    parser.add_argument('-c', '--config_file', type=str, metavar="", default='configs.json')
    parser.add_argument('-i', '--gpu_id', type=str, metavar="", default=0)
    parser.add_argument('-t', '--tree_method', type=str, metavar="", default='gpu_hist')
    parser.add_argument('-b', '--business', type=str, metavar="", default='huabei')
    parser.add_argument('-g', '--granularity', type=int, metavar="", default=5)
    parser.add_argument('-p', '--predict_size', type=int, metavar="", default=5)
    parser.add_argument('-e', '--event', type=str, metavar="", default='1204')
    parser.add_argument('-v', '--call_inflow', type=str, metavar="", default='20191118')
    # Parse args.
    args = parser.parse_args()
    # Namespace => Dictionary.
    kwargs = vars(args)

    return Config(**kwargs)
