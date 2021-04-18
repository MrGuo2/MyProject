# -*- coding: utf-8 -*-
from configs import get_configs
from data_preprocessing import DataPreprocess
from utils.log import logger
from xgboost_model import XgboostGridSearchModel
from da_rnn_model import DaRnnModel
from tcn_model import TcnModel

SEPARATOR = "\n" + "*" * 75 + "Sperator" + "*" * 75


def main():
    logger.info(SEPARATOR)
    configs = get_configs()

    # Run data preprocess
    if configs.data_preprocess_active:
        logger.info(configs)
        preprocess = DataPreprocess(configs)
        preprocess.data_preprocess()
        logger.info("Data preprocess finished!")
    # Run train model
    if configs.da_rnn_model_active:
        logger.info(configs)
        da_rnn_model = DaRnnModel(configs)
        da_rnn_model.run()
        logger.info("Da_rnnModel finished!")
    if configs.xgboost_gridsearch_model_active:
        xgboost_model = XgboostGridSearchModel(configs)
        xgboost_model.run()
        logger.info("XGboost finished!")
    if configs.tcn_big_file_model_active:
        tcn_model = TcnModel(configs)
        tcn_model.run()
        logger.info("TcnModel finished!")


if __name__ == '__main__':
    main()
