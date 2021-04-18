from torch import nn
from torch import optim
from modules import Encoder, Decoder
import requests
from custom_types import DaRnnNet, TrainConfig
from utils.file import *
from utils.log import logger
from torch.utils.tensorboard import SummaryWriter

FORMAT_PATTERN = '%Y-%m-%d %H:%M'


class DaRnnModel:
    def __init__(self, config):
        self.data_dir_path = config.data_dir_path
        self.da_rnn_kwargs = config.da_rnn_model["params"]
        self.device = torch.device(config.da_rnn_model["params"]["device_gpu"])
        logger.info(f"Using computation device: {self.device}")
        for key, value in config.da_rnn_model.items():
            setattr(self, key, value)
        for key, value in config.data_preprocess.items():
            setattr(self, key, value)

    def get_data(self, input_file, debug):
        dataset = pd.read_csv(input_file, nrows=4000 if debug else None)
        train_data = dataset.iloc[:, 2:]
        label_data = dataset.iloc[:, 1]
        return np.array(train_data), np.expand_dims(label_data, axis=1)

    def show_acc_for_ding(self, dd_url, content):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        post_data = {"msgtype": "text",
                     "text": {
                         "content": content},
                     "at": {
                         "atMobiles": self.info_phone}
                     }
        requests.post(dd_url, headers=headers, data=json.dumps(post_data))

    def da_rnn_model(self, train_data, train_label, evaluate_file, device, **da_rnn_kwargs):

        def predict(params, train_data, train_label, configs, on_train, device):
            out_size = train_label.shape[1]
            if on_train:
                y_pred = np.zeros((train_data.shape[0] - configs.time_step, train_label.shape[1]))
            else:
                y_pred = np.zeros((train_data.shape[0] - configs.train_size - self.predict_size, out_size))

            for y_i in range(0, len(y_pred), configs.batch_size):
                y_slc = slice(y_i, y_i + configs.batch_size)
                batch_idx = range(len(y_pred))[y_slc]
                b_len = len(batch_idx)
                train_set = np.zeros((b_len, int(configs.time_step / self.predict_size),
                                      train_data.shape[1]))
                label_set = np.zeros((b_len, int(configs.time_step / self.predict_size),
                                      train_label.shape[1]))

                for n_batch, batch_index in enumerate(batch_idx):
                    if on_train:
                        idx = range(batch_index, batch_index + configs.time_step, self.predict_size)
                    else:
                        idx = range(batch_index + train_size - configs.time_step, batch_index + train_size,
                                    self.predict_size)

                    train_set[n_batch, :, :] = train_data[idx, :]
                    label_set[n_batch, :] = train_label[idx]

                label_set = numpy_to_tvar(label_set, device)
                _, input_encoded = params.encoder(numpy_to_tvar(train_set, device))
                y_pred[y_slc] = params.decoder(input_encoded, label_set).cpu().data.numpy()

            return y_pred

        def set_params(train_data, device, **da_rnn_kwargs):
            train_configs = TrainConfig(da_rnn_kwargs["time_step"],
                                        int(train_data.shape[0] * 0.95),
                                        da_rnn_kwargs["batch_size"],
                                        nn.MSELoss())

            enc_kwargs = {"input_size": train_data.shape[1],
                          "hidden_size": da_rnn_kwargs["en_hidden_size"],
                          "time_step": int(da_rnn_kwargs["time_step"] / self.predict_size)}
            dec_kwargs = {"encoder_hidden_size": da_rnn_kwargs["en_hidden_size"],
                          "decoder_hidden_size": da_rnn_kwargs["de_hidden_size"],
                          "time_step": int(da_rnn_kwargs["time_step"] / self.predict_size),
                          "out_feats": da_rnn_kwargs["target_cols"]}
            encoder = Encoder(**enc_kwargs).to(device)
            decoder = Decoder(**dec_kwargs).to(device)

            encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad],
                                           lr=da_rnn_kwargs["learning_rate"], betas=(0.9, 0.999), eps=1e-08)
            decoder_optimizer = optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad],
                                           lr=da_rnn_kwargs["learning_rate"], betas=(0.9, 0.999), eps=1e-08)
            da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

            return train_configs, da_rnn_net

        # Preprocess train data
        def prep_train_data(train_iter, configs, train_data, train_label):
            # Generator fit and evaluate set in batch_count
            data_set = np.zeros((len(train_iter), int(configs.time_step / self.predict_size),
                                 train_data.shape[1]))
            label_set = np.zeros((len(train_iter), int(configs.time_step / self.predict_size),
                                  train_label.shape[1]))
            test_set = train_label[train_iter + configs.time_step + self.predict_size]
            for batch_index, per_line in enumerate(train_iter):
                batch_slice = slice(per_line, per_line + configs.time_step, self.predict_size)
                data_set[batch_index, :, :] = train_data[batch_slice, :]
                label_set[batch_index, :] = train_label[batch_slice]

            return data_set, label_set, test_set

        # Generate loss valuate
        def loss_func(net_params, loss_func, train_data, train_label, test_label, device):
            net_params.enc_opt.zero_grad()
            net_params.dec_opt.zero_grad()
            input_weighted, input_encoded = net_params.encoder(numpy_to_tvar(train_data, device))
            pred_label = net_params.decoder(input_encoded, numpy_to_tvar(train_label, device))
            true_label = numpy_to_tvar(test_label, device)
            loss = loss_func(pred_label, true_label)
            loss.backward()
            net_params.enc_opt.step()
            net_params.dec_opt.step()

            return loss.item()

        def predict_acc(params, configs, evaluate_file, debug, device, **da_rnn_kwargs):
            train_data, train_label = self.get_data(evaluate_file, debug)
            final_y_pred = predict(params, train_data, train_label, configs, on_train=True, device=device)
            final_y_target = train_label[range(da_rnn_kwargs["time_step"], train_label.shape[0])]
            acc = model_accuracy(np.squeeze(final_y_pred), np.squeeze(final_y_target))
            print(acc)
            return acc

        configs, params = set_params(train_data, device, **da_rnn_kwargs)
        # Generator batch_count and show fit info
        batch_count = int(np.ceil(configs.train_size * 1. / configs.batch_size))
        logger.info(f"Iterations per epoch: {configs.train_size * 1. / configs.batch_size:3.3f} ~ {batch_count:d}.")
        logger.info(f"Train size is: {configs.train_size}")
        logger.info(f"Batch size is: {configs.batch_size}")
        logger.info(f"Device is: {device}")
        # Generator container for iter_loss&epoch_loss
        epoch_loss = np.zeros(da_rnn_kwargs["epoch"])
        iter_loss = np.zeros(da_rnn_kwargs["epoch"] * batch_count)
        n_iter = 0
        best_acc = 0.0
        writer = SummaryWriter()

        for n_epoch, epoch_id in enumerate(range(da_rnn_kwargs["epoch"])):
            train_size = configs.train_size - configs.time_step - self.predict_size
            iter_train = np.random.permutation(train_size)
            for iter_id in range(0, train_size, configs.batch_size):
                train_iter = iter_train[iter_id: (iter_id + configs.batch_size)]
                train_set, label_set, test_set = prep_train_data(train_iter, configs, train_data, train_label)
                loss = loss_func(params, configs.loss_func, train_set, label_set, test_set, device)
                iter_loss[epoch_id * batch_count + iter_id // configs.batch_size] = loss
                # Show per batch_count loss
                if iter_id % 500 == 0:
                    logger.info("Epoch %d, Batch %d: loss = %3.3f.", epoch_id, n_iter, loss)
                writer.add_scalar("Loss/iterator_loss", loss, n_iter+1)
                n_iter += 1
            epoch_loss[epoch_id] = np.mean(iter_loss[range(epoch_id * batch_count, (epoch_id + 1) * batch_count)])
            # Show per epoch loss
            if n_epoch % 1 == 0:
                y_test_pred = predict(params, train_data, train_label, configs, on_train=False, device=device)
                val_loss = y_test_pred - train_label[configs.train_size + self.predict_size:]
                logger.info(f"Epoch {epoch_id}, loss: {epoch_loss[epoch_id]:3.3f}, loss: {np.mean(np.abs(val_loss))}.")
                acc = predict_acc(params, configs, evaluate_file, self.debug, device, **da_rnn_kwargs)
                if acc > best_acc:
                    best_acc = acc
                    # Save model
                    torch.save(params.encoder.state_dict(), self.encoder_model)
                    torch.save(params.decoder.state_dict(), self.decoder_model)
                content = f"Out of epoch: {epoch_id} result！\n" \
                          f"{self.business_line}accuracy is: {acc}\n" \
                          f"The best model accuracy is : {best_acc}"
                self.show_acc_for_ding(self.dd_url, content)
        content = f"Da_rnn model is Finished!\n" \
                  f"The best model accuracy is : {best_acc}\n" \
                  f"Details:{self.encoder_model}"
        self.show_acc_for_ding(self.dd_url, content)

    def run(self):
        # Read train data
        fit_file = os.path.join(self.data_dir_path, self.fit_slide_window)
        evaluate_file = os.path.join(self.data_dir_path, self.evaluate_slide_window)
        train_data, label_data = self.get_data(fit_file, self.debug)
        self.da_rnn_model(train_data, label_data, evaluate_file, self.device, **self.da_rnn_kwargs)
