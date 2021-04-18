# -*- coding: utf-8 -*-
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from utils.file import *
from utils.log import logger
import pandas as pd

FORMAT_PATTERN = '%Y-%m-%d %H:%M'


class DataPreprocess(object):
    def __init__(self, config):
        self.data_dir_path = config.data_dir_path
        self.input_file_name = config.input_file
        self.merge_file_dir_path = config.merge_file_dir_path
        self.data_preprocess_active = config.data_preprocess_active
        self.one_day = config.data_preprocess["one_day"]
        for key, value in config.data_preprocess.items():
            setattr(self, key, value)
        for key, value in config.tcn_model.items():
            setattr(self, key, value)
        for key, value in config.da_rnn_model.items():
            setattr(self, key, value)

    def data_preprocess(self):
        # Download data path
        stitching_file_name = os.path.join(self.data_dir_path, self.merge_file_name)
        del_other_file = os.path.join(self.data_dir_path, self.del_other_file)
        per_min_fit_file = os.path.join(self.data_dir_path, self.per_min_fit_file)
        per_min_evaluate_file = os.path.join(self.data_dir_path, self.per_min_evaluate_file)
        events_file = os.path.join(self.data_dir_path, self.events_file)
        fit_events = os.path.join(self.data_dir_path, self.fit_events)
        evaluate_events = os.path.join(self.data_dir_path, self.evaluate_events)
        fit_slide_window = os.path.join(self.data_dir_path, self.fit_slide_window)
        evaluate_slide_window = os.path.join(self.data_dir_path, self.evaluate_slide_window)
        fit_file = os.path.join(self.data_dir_path, self.fit_file)
        evaluate_file = os.path.join(self.data_dir_path, self.evaluate_file)
        fit_npz_file = os.path.join(self.data_dir_path, self.fit_npz_file)
        evaluate_npz_file = os.path.join(self.data_dir_path, self.evaluate_npz_file)

        # Stitching './data/to_be_merge'files
        if self.stitching_file:
            self.data_stitching(self.merge_file_dir_path, stitching_file_name)
            logger.info("Stitching file finished!")
        # Delete other feature
        if self.other_feature:
            self.delete_other_feature(stitching_file_name, del_other_file)
        # Generate Calls per minute
        if self.per_minute:
            self.sort_per_minute(del_other_file, per_min_fit_file, per_min_evaluate_file)
            logger.info("Generate per minute data finished!")
            per_min_fit = pd.read_csv(per_min_fit_file, header=None)
            per_min_evaluate = pd.read_csv(per_min_evaluate_file, header=None)
            logger.info(f"\nThe first five lines 'per_min_fit' data is:\n{per_min_fit.head()}")
            logger.info(f"\nThe first five lines 'per_min_evaluate' data is:\n{per_min_evaluate.head()}")
        # event processing
        if self.presence_event:
            self.event_processing(events_file, fit_events, evaluate_events)
            logger.info("event processing finished!")
        # Generate Rolling data
        if self.merge_file:
            self.create_features(per_min_fit_file, fit_events, fit_slide_window, self.granularity)
            self.create_features(per_min_evaluate_file, evaluate_events, evaluate_slide_window, self.granularity)
            logger.info("Generate slide window finished!")
        if self.add_feature:
            self.add_feats(fit_slide_window, fit_slide_window)
            self.add_feats(evaluate_slide_window, evaluate_slide_window)
            logger.info("Add features finished!")
        if self.generate_data_csv:
            self.generator_data(fit_slide_window, fit_file, self.granularity, self.predict_size)
            self.generator_data(evaluate_slide_window, evaluate_file, self.granularity, self.predict_size)
        if self.generate_data_npz:
            self.get_npz_file(fit_slide_window, fit_npz_file, self.predict_size, self.granularity)
            self.get_npz_file(evaluate_slide_window, evaluate_npz_file, self.predict_size, self.granularity)
            logger.info("Generate npz file finished!")

    def delete_other_feature(self, input_file, output_file):
        '''Filter data for corresponding characteristics

        :param input_file: Input three feature file ['index', 'Dtime', 'business']
        :param output_file: Output three feature file ['index', 'Dtime', 'business']
        :return:
        '''
        data_frame = pd.read_csv(input_file, engine='python', header=None)
        new_data_frame = data_frame.loc[data_frame[2].isin([self.business_line])]
        new_data_frame.to_csv(output_file, header=None, index=None)

    def get_time_index(self, input_data_frame, start_time, end_time):
        '''Split file into fit and evaluate set

        :param input_data_frame: Input DataFrame
        :return: Two list
        '''
        start_time = input_data_frame[input_data_frame[0] == start_time].index[0]
        end_time = input_data_frame[input_data_frame[0] == end_time].index[0]

        fit_list = pd.concat([input_data_frame[:start_time], input_data_frame[end_time:]], axis=0)
        evaluate_list = input_data_frame[start_time:end_time]
        return fit_list, evaluate_list

    def sort_per_minute(self, input_file, output_fit_file, output_evaluate_file):
        '''Sort and class the files

        :param input_file: Input three features file ['Index', 'Dtime', 'Business']
        :param output_fit_file: Output two features file ['Dtime', 'frequency']
        :param output_evaluate_file: Output two features file ['Dtime', 'frequency']
        '''
        data_frame = pd.read_csv(input_file, header=None, usecols=list(range(3)))
        data_frame['Dtime'] = data_frame[1].apply(lambda x: datetime.strptime(x[:-3], FORMAT_PATTERN))
        data_frame = data_frame.groupby(['Dtime']).agg("size")  # as_index(default:True)
        # Make new index for "data_time"
        # Creat a new index
        min_ts = data_frame.index.min()
        max_ts = data_frame.index.max()
        date_list = pd.date_range(min_ts, max_ts, freq='1Min')
        new_index = pd.DataFrame(date_list, columns=['Dtime'])
        data_frame = data_frame.reset_index(drop=False)
        new_list = pd.merge(new_index, data_frame, on=['Dtime'], how="outer").fillna(0)
        new_list.columns = [0, 1]
        fit_data_frame, evaluate_data_frame = self.get_time_index(new_list, self.start_time, self.end_time)
        fit_data_frame.to_csv(output_fit_file, encoding='utf-8', header=None, index=None)
        evaluate_data_frame.to_csv(output_evaluate_file, encoding='utf-8', header=None, index=None)

    def event_processing(self, events_file, fit_events, evaluate_events):
        '''Sort and preprocess events files

        :param events_file: Input N feature file ['Dtime', 'other_feature'...]
        :param fit_events: Output N feature file ['Dtime', 'other_feature'...]
        :param evaluate_events: Output N feature file ['Dtime', 'other_feature'...]
        :return:
        '''
        events_dataframe = pd.read_csv(events_file, header=None)
        events_dataframe[events_dataframe == ""] = np.NaN
        sort_events = events_dataframe.groupby([0], as_index=False).sum()
        sort_events.sort_values(by=[0], inplace=True)
        sort_events[0] = sort_events[0].apply(lambda ss: datetime.strptime(ss[:-3], FORMAT_PATTERN))

        min_ts = sort_events[0].min()
        max_ts = sort_events[0].max()
        date_list = pd.date_range(min_ts, max_ts, freq='1Min')
        new_index = pd.DataFrame(date_list)
        new_dataframe = pd.merge(new_index, sort_events, how="outer").fillna(0)

        def precess_events(events_dataframe):
            '''Normalize and Select the data

            :param events_dataframe: Input a data frame ['Dtime', 'other_features'...]
            :return: Output a data frame ['Dtime', 'other_features'...]
            '''
            minmax_scaler = MinMaxScaler()
            other_features = minmax_scaler.fit_transform(events_dataframe.loc[:, 1:])
            time_stamp = np.expand_dims(list(map(str, events_dataframe[0])), axis=1)
            features = pd.DataFrame(np.concatenate([time_stamp, other_features], axis=1))

            if self.business == "huabei":
                features = pd.concat([features.loc[:, 0:9], features.loc[:, 19:27], features.loc[:, 37:45],
                                      features.loc[:, 55:63], features.loc[:, 73]], axis=1, ignore_index=True)
            return features

        np.expand_dims(list(map(str, events_dataframe[0])), axis=1)
        features = precess_events(new_dataframe)
        # Split a file into two files that fit and evaluate file
        fit_data_frame, evaluate_data_frame = self.get_time_index(features, self.start_time, self.end_time)
        fit_data_frame.to_csv(fit_events, encoding='utf-8', header=None, index=None)
        evaluate_data_frame.to_csv(evaluate_events, encoding='utf-8', header=None, index=None)

    def create_features(self, input_permin, input_events, output_slide_window, granularity):
        '''Generate slide window file

        :param input_permin:
        :param input_features:
        :param output_slide_window:
        :param granularity:
        :return:
        '''
        if self.presence_event:
            features = merge_file(input_permin, input_events)
        else:
            features = get_file_content(input_permin)

        write_slide_window = open(output_slide_window, 'w')
        diff = str(datetime.strptime("00:05", '%H:%M') + timedelta(minutes=granularity)).split(' ')[1].split(':')
        for i in range(len(features) - granularity):
            feature = features[i].split(',')
            temp_status = []
            for temp_add in features[i:(i + granularity)]:
                other_feature = temp_add.split(',')[1:]
                other_feature = list(map(float, other_feature))
                temp_status.append(other_feature)
            line_feature = [feature[0], ] + list(map(str, np.array(temp_status).sum(axis=0).tolist()))
            hour = int(line_feature[0].split(' ')[-1].split(':')[0])
            minute = int(line_feature[0].split(' ')[-1].split(':')[1])
            morning_diff = (hour == int(diff[0]) and minute >= int(diff[1])) or (hour > int(diff[0]))
            if self.working_hours and (hour < 8 and morning_diff):
                continue
            write_slide_window.write(','.join(line_feature) + '\n')
        write_slide_window.close()

    def add_feats(self, input_file, output_file):
        input_file_df = pd.read_csv(input_file, header=None)
        date_series = input_file_df[0]
        add_new_feats = []
        for date_line in date_series:
            day = date_line.split(" ")[0].split("-")[-1]
            hour = date_line.split(" ")[-1].split(":")[0]
            date_line = datetime.strptime(date_line, "%Y-%m-%d %H:%M:%S")
            week = date_line.isoweekday()
            add_new_feats.append([day, hour, week])
        add_feats_df = pd.DataFrame(add_new_feats)
        output_df = pd.concat([input_file_df, add_feats_df], axis=1)
        output_df.to_csv(output_file, header=None, index=None)

    def generator_data(self, input_file, output_filepath, granularity, predict_size):

        def array_to_str(array, seperator):
            return seperator.join([str(arr) for arr in array])

        def predict_data(granularity, data, predict_size):
            if granularity == predict_size:
                time_y = data.split(',')[0]
                y = float(data.split(",")[1])
                return y, time_y
            else:
                sum_y = []
                time_y = data[0].split(',')[0]
                for i in data[::granularity]:
                    i = float(i.split(',')[1])
                    sum_y.append(i)
                return sum(sum_y), time_y

        def gen_slide_window(data, granularity, ties):
            diff = str(datetime.strptime("00:05", '%H:%M') + timedelta(minutes=granularity)).split(' ')[1].split(':')
            new_data = []
            for line in data:
                hour = int(line.split(',')[0].split(' ')[-1].split(':')[0])
                minute = int(line.rstrip('\n').split(',')[0].split(' ')[-1].split(':')[1])
                morning_diff = (hour == int(diff[0]) and minute >= int(diff[1])) or (hour > int(diff[0]))
                if self.working_hours and (hour < 8 and morning_diff):
                    continue
                new_data.append(line)

            x, time_x = [], []
            for i in range(0, len(new_data), granularity):
                if ties:
                    features = list(map(float, new_data[i].split(',')[1:]))
                else:
                    features = float(new_data[i].split(',')[1])
                time_x.append(new_data[i].split(',')[0])
                x.append(features)
            return x, time_x

        data = get_file_content(input_file)
        diff = str(datetime.strptime("00:05", '%H:%M') + timedelta(minutes=granularity)).split(' ')[1].split(':')
        slide_window_data = get_file_content(input_file)
        with open(output_filepath, "w", encoding='utf-8') as f:
            for index, i in enumerate(range(self.one_day, len(slide_window_data) - predict_size)):
                hour = int(slide_window_data[i].rstrip('\n').split(',')[0].split(' ')[-1].split(':')[0])
                minute = int(slide_window_data[i].rstrip('\n').split(',')[0].split(' ')[-1].split(':')[1])
                morning_diff = (hour == int(diff[0]) and minute >= int(diff[1])) or (hour > int(diff[0]))
                if self.working_hours and (hour < 8 and morning_diff):
                    continue

                ties = len(slide_window_data[0].split(',')) - 2
                x, time_x = gen_slide_window(slide_window_data[i - self.one_day: i], granularity, ties)
                if granularity == predict_size:
                    y, time_y = predict_data(granularity, data[i + 5], predict_size)
                else:
                    y, time_y = predict_data(granularity, data[i + 5: i + predict_size + 5], predict_size)

                if ties:
                    x_line = array_to_str([array_to_str(arr, ',') for arr in x], '\t')
                    y_line = str(y)
                    time_x_line = array_to_str(time_x, ',')
                    time_y_line = str(time_y)
                else:
                    x_line = array_to_str(x, '\t')
                    y_line = str(y)
                    time_x_line = array_to_str(time_x, ',')
                    time_y_line = str(time_y)
                f.write('#'.join([x_line, y_line, time_x_line, time_y_line]) + '\n')

    def get_npz_file(self, input_file, output_file, predict_size, granularity):

        def gen_slide_window(data_set, granularity, information):

            x, time_x = [], []
            if information:
                for i in range(0, len(data_set), granularity):
                    features = list(map(float, data_set[i].split(',')[1:]))
                    time_x.append(data_set[i].split(',')[0])
                    x.append(features)
            else:
                for i in range(0, len(data_set), granularity):
                    feature = float(data_set[i].split(',')[1])
                    time_x.append(data_set[i].split(',')[0])
                    x.append(feature)

            return x, time_x

        def predict_data(granularity, data):
            if granularity == predict_size:
                time_y = data.split(',')[0]
                y = float(data.split(",")[1])
                return y, time_y
            else:
                sum_y = []
                time_y = data[0].split(',')[0]
                for i in data[::granularity]:
                    i = float(i.split(',')[1])
                    sum_y.append(i)
                return sum(sum_y), time_y

        list_x, list_y, list_time_x, list_time_y = [], [], [], []
        diff = str(datetime.strptime("00:05", '%H:%M') + timedelta(minutes=granularity)).split(' ')[1].split(':')
        data = get_file_content(input_file)

        for i in tqdm(list(range(self.one_day, len(data) - predict_size))):
            hour = int(data[i].rstrip('\n').split(',')[0].split(' ')[-1].split(':')[0])
            minute = int(data[i].rstrip('\n').split(',')[0].split(' ')[-1].split(':')[1])
            morning_diff = (hour == int(diff[0]) and minute >= int(diff[1])) or (hour > int(diff[0]))
            ties = len(data[0].split(",")) - 2
            if self.working_hours and (hour < 8 and morning_diff):
                continue
            x, time_x = gen_slide_window(data[i - self.one_day:i], granularity, ties)

            if granularity == predict_size:
                y, time_y = predict_data(granularity, data[i + 5])
            else:
                y, time_y = predict_data(granularity, data[i + 5: i + predict_size + 5])
            list_x.append(x)
            list_y.append(y)
            list_time_x.append(time_x)
            list_time_y.append(time_y)

        np.savez(output_file, list_x, list_y, list_time_x, list_time_y)
