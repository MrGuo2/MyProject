import numpy as np 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 
import os

data_path = "/home/wb-ghd629671/sequence_prediction/huabei/temp_file/"
output_plot_path = "/home/wb-ghd629671/sequence_prediction/huabei/temp_code/"


def check_feature(input_filepath, output_filepath):
    plt.figure(figsize=(160, 15))
    font1 = {'weight': 'normal', 'size': 55}
    for input_file in os.listdir(input_filepath):
        read_file = pd.read_table(input_filepath + input_file, header=None, delimiter=' ')
        read_file[0] = read_file[0] + " " + read_file[1]
        read_file = read_file.drop([1], axis=1)
        read_file[0].apply(lambda ss: datetime.strptime(str(ss)[:-3], '%Y-%m-%d %H:%M'))
        plt.plot(read_file[0], read_file[2], label=input_file)

    plt.legend(prop=font1)
    plt.xlabel('Time')
    plt.ylabel('label')
    plt.xticks(range(0, 4500, 10), rotation=70)
    plt.savefig(output_filepath + "test.png", bbox_inches="tight")


def get_int(input_filepath):
    plt.figure(figsize=(550, 15))
    font1 = {'weight': 'normal', 'size': 55}
    for input_file in os.listdir(input_filepath):
        read_file = pd.read_csv(input_filepath + input_file, header=None)
        plt.plot(read_file[0], read_file[1], label="PredictLabel")
        plt.plot(read_file[0], read_file[2], label="TrueLabel")

    plt.legend(prop=font1)
    plt.xlabel('Time')
    plt.ylabel('label')
    plt.xticks(range(0, 45000, 10), rotation=70)
    plt.savefig(output_plot_path + "test.png", bbox_inches="tight")


if __name__ == '__main__':
    # check_feature(data_path, output_plot_path)
    get_int(data_path)