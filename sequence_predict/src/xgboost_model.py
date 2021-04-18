# encoding: utf-8
from utils.file import *
from utils.log import logger
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


class XgboostGridSearchModel(object):

    def __init__(self, config):
        for key, value in config.xgboost_model.items():
            setattr(self, key, value)
        for prep_key, prep_value in config.data_preprocess.items():
            setattr(self, prep_key, prep_value)

        self.data_dir_path = config.data_dir_path
        self.input_file_path = config.input_file

    def run(self):
        train_data, train_time, train_label, test_data,\
            test_time, test_label = self.get_feature(self.fit_npz_file, self.evaluate_npz_file)
        self.train_xgboost_gridsearch(train_data, train_label, test_data, test_label, train_time, test_time)

    def get_feature(self, train_name, test_name):
        '''
        Add data to be added
        Args:
            trianDatas:  .npz files
            test_name:  .npz files
        Returns:

        '''
        # Read train and test data
        input_train_file = np.load(os.path.join(self.data_dir_path, train_name))
        input_test_file = np.load(os.path.join(self.data_dir_path, test_name))
        train_data, train_label, train_time = \
            input_train_file["arr_0"], input_train_file["arr_1"], input_train_file["arr_2"]
        test_data, test_label, test_time = \
            input_test_file["arr_0"], input_test_file["arr_1"], input_test_file["arr_2"]

        # Checking test data
        logger.info("\nChecking test data")
        logger.info(f"\nThe test first data is: \n{test_data[0]}\nThe data shape is: {test_data.shape}")
        logger.info(f"\nThe test first label is: \n{test_label[0]}\nThe data shape is: {test_label.shape}")

        return train_data, train_time, train_label, test_data, test_time, test_label

    def train_xgboost_gridsearch(self, train_data, train_label, test_data, test_label, train_time, test_time):
        '''Model application
        Args:
            train_data: Train_data
            train_label: Train_label
            test_data: Test_data
            test_label: Test_label
            train_time: Train_time
            test_time: Test_time
        '''

        xlf = xgb.XGBRegressor(self.grid_search_params)
        if self.adjust_params:
            grid_params = dict(self.const_params, **self.params)
            grid_search_train = GridSearchCV(xlf, param_grid=grid_params, verbose=2)
            grid_result_label = grid_search_train.fit(train_data, train_label)
        else:
            grid_params = {k: [v[0]] for k, v in self.params.items()}
            grid_params = dict(self.const_params, **grid_params)
            grid_search_train = GridSearchCV(xlf, param_grid=grid_params, verbose=2)
            grid_result_label = grid_search_train.fit(train_data, train_label)

        show_bst = grid_result_label.best_estimator_
        show_predict = show_bst.predict(test_data)
        print("The best score is:{:.3f}".format(r2_score(test_label, show_predict)))
        logger.info(f"Best accuracy on training set:{grid_result_label.best_score_}")
        logger.info(f"Best accuracy on testing set:{model_accuracy(test_label, show_predict)}")

        # Get best parameters
        best_parameters = show_bst.get_params()
        for param_name in sorted(self.params.keys()):
            logger.info(f"\t{param_name}: {best_parameters[param_name]}")

        # Output everyday accuracy
        # todo Need to fix bug
        if self.everyday_accuracy:
            everyday_accuracy_print(test_data, test_label, test_time, self.working_hours, show_bst,
                                    self.everyday_accuracy_path)
            logger.info(f"The path to generate the xgboost everyday accuracy is: {self.everyday_accuracy_path}.")
        # Visualization of prediction results
        if self.output_plot:
            plot_pred(show_predict, test_label, test_time, self.plot_path_name)
            logger.info(f"The path to generate the xgboost image is: {self.plot_path_name}.")
        # Print of prediction results
        if self.output_print_result:
            print_pred(show_predict, test_label, test_time, self.print_path_name)
            logger.info(f"The path to generate the xgboost diff_file is: {self.print_path_name}.")
        # Save model
        if self.output_model:
            show_bst.save_model(self.model_dir_path)
            logger.info(f"The path to generate the xgboost model is: {self.model_dir_path}.")
        # Print feature importance
        # todo 'XGBRegressor' object has no attribute 'get_fscore' we must write new function
        if self.output_feature_importance:
            feature_score = show_bst.get_fscore()
            feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
            fs = []
            for (key, value) in feature_score:
                fs.append("{0},{1}\n".format(key, value))
            with open(os.path.join(self.data_dir_path, self.feature_importance), 'w') as f:
                f.writelines("feature,score\n")
                f.writelines(fs)
            logger.info(f"The path to generate the xgboost feature importance is: {self.feature_importance}.")
