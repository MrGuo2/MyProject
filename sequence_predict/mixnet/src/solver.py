from collections import defaultdict
import os

import numpy as np
from pytorch_pretrained_bert import BertAdam
import torch
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.loss import *
from modules.metrics import *
from modules.rlloss import *
from utils.log import logger
from utils.time_track import time_desc_decorator


class MultipleOptimizer(object):
    """ Class for multiple optimizers."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class Solver(object):
    """Class for training and evaluating."""

    def __init__(self, config, train_data_loader, valid_data_loader, feature_index, model, device, RS='Supervised'):
        logger.info('Init Solver.')
        self.config = config
        self.device = device
        self.feature_index = feature_index
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.RS = RS

        self.current_epoch = 1
        self.train_global_step = 0
        self.valid_global_step = 0

    @time_desc_decorator('building Solver')
    def build(self):
        """Building model."""
        if self.config.action == 'train':
            logger.info(f'Init TensorBoard. Save path: {self.config.tb_log_dir}.')
            self.writer = SummaryWriter(self.config.tb_log_dir)
            # TODO: add model to TensorBoard.
            # self.writer.add_graph(self.model)
            self.optimizer = MultipleOptimizer(self.__get_optimizer())

        if self.config.checkpoint:
            checkpoint_path = os.path.join(self.config.model_dir, self.config.checkpoint)
            assert os.path.exists(checkpoint_path), f'Checkpoint file {checkpoint_path} does not exist.'
            self.load_model(checkpoint_path)
            # checkpoint_path = os.path.join(self.config.model_dir, self.config.checkpoint)
            # self.load_model(checkpoint_path)

        logger.info(f'Put the model to {self.device}.')
        self.model.to(self.device)

        logger.info('=========================Model Parameters=========================')
        param_count = 0
        for name, param in self.model.named_parameters():
            param_count += param.numel()
            logger.info(f'{name}: {list(param.size())}')
        logger.info(f'Model parameters count: {param_count:,}.')

        self.__build_loss_weight()

    def __build_loss_weight(self):
        """Building loss weight from output config."""
        logger.info('Building loss weight from output config.')
        self.label_loss_weight = {}
        self.neg_loss_weight = {}
        for label_name, output_config in self.config.output_config_dict.items():
            label_weight = output_config.get('weight', 1)
            self.label_loss_weight[label_name] = torch.tensor(label_weight, dtype=torch.float, device=self.device)
            neg_weight = output_config.get('neg_loss_weight', 1)
            self.neg_loss_weight[label_name] = torch.tensor(neg_weight, dtype=torch.float, device=self.device)

    def __get_optimizer(self):
        """Init optimizer."""
        logger.info(f'Init optimizer {self.config.optimizer}.')
        # Get memory addresses of bert parameters.
        bert_param_ids = set()
        for model_name, model_config in self.config.model_config_dict.items():
            if model_config['type'] == 'bert':
                bert_model = self.model.subnet[model_name]
            elif model_config['type'] == 'mask_self_attn' and 'bert' in model_config['params']['inputs']:
                bert_model = self.model.subnet[model_name].subnet['bert']
            else:
                continue
            for param in bert_model.parameters():
                bert_param_ids.add(id(param))

        # Split model parameters into base parameters and bert parameters.
        base_params = []
        bert_name_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if id(param) in bert_param_ids:
                    bert_name_params.append((name, param))
                else:
                    base_params.append(param)
        # Add optimizer for base parameters.
        optimizers = []
        learning_rate = self.config.learning_rate
        base_optimizer_name = self.config.optimizer.lower()
        if base_optimizer_name == 'adam':
            base_optimizer = optim.Adam(base_params, lr=learning_rate)
        elif base_optimizer_name == 'sgd':
            base_optimizer = optim.SGD(base_params, lr=learning_rate, momentum=self.config.momentum)
        elif base_optimizer_name == 'adadelta':
            base_optimizer = optim.Adadelta(base_params, lr=learning_rate)
        else:
            raise NotImplementedError(f'The optimizer {base_optimizer_name} is not implemented.')
        optimizers.append(base_optimizer)
        # Add optimizer for bert parameters.
        if self.config.use_bert:
            bert_learning_rate = self.config.bert_learning_rate
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_grouped_params = [
                {'params': [p for n, p in bert_name_params if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in bert_name_params if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            total_batch = len(self.train_data_loader.dataset) * self.config.epoch_num
            bert_optimizer = BertAdam(bert_grouped_params,
                                      lr=bert_learning_rate,
                                      warmup=self.config.bert_warmup_prop,
                                      t_total=total_batch)
            optimizers.append(bert_optimizer)

        return optimizers

    def load_model(self, checkpoint):
        """Load model from checkpoint.

        Args:
            checkpoint: str, checkpoint path.

        Returns:
            None.
        """
        logger.info(f'Loading model from {checkpoint}.')
        epoch = int(os.path.basename(checkpoint)[6:-3])
        self.current_epoch = epoch + 1
        self.model = torch.load(checkpoint, map_location=self.device)
        if self.current_epoch > self.config.epoch_num:
            logger.warn(f'The current epoch {self.current_epoch} > total epoch number {self.config.epoch_num}.')

    def save_model(self, epoch):
        """Save model to checkpoint.

        Args:
            epoch: int, epoch num.

        Returns:
            None.
        """
        model_name = f'model_{epoch}.pt'
        model_path = (os.path.join(self.config.model_dir, model_name))
        torch.save(self.model, model_path)
        logger.info(f'The model has been saved in {model_path}.')

    def print_log(self, label_name, global_step=None):
        """Print log according to label name and global step.

        If global_step is None, the method will print epoch-level loss and metrics. Otherwise, the method prints
        batch-level loss and metrics.

        The variables to print are:
            * self.loss: Weighted multiplication of self.pos_loss and self.neg_loss.
            * self.pos_loss: Positive loss.
            * self.neg_loss: Negative loss.
            * self.metrics_dict: Metrics of positive samples.
            * self.neg_metrics_dict: Metrics of negative samples.

        Args:
            label_name: str, label name.
            global_step: int or None, global batch step.

        Returns:
            None.
        """
        label_name_cap = label_name.capitalize()
        if global_step is None:
            logger.info(f'{label_name_cap} epoch step: {self.current_epoch}, total loss: {self.loss:.2f}, '
                        f'positive loss: {self.pos_loss:.2f}, negative loss: {self.neg_loss:.2f}.')
        elif global_step % self.config.log_interval == 0:
            logger.info(f'{label_name_cap} batch step: {global_step}, total loss: {self.loss:.2f}, '
                        f'positive loss: {self.pos_loss:.2f}, negative loss: {self.neg_loss:.2f}.')
        else:
            return
        pos_metrics_list = [f'{name}: {value:.4f}' for name, value in self.metrics_dict.items()]
        neg_metrics_list = [f'{name}: {value:.4f}' for name, value in self.neg_metrics_dict.items()]
        logger.info(f'{label_name_cap} positive metrics: {", ".join(pos_metrics_list)}.')
        logger.info(f'{label_name_cap} negative metrics: {", ".join(neg_metrics_list)}.')

    def write_summary(self, prefix, global_step):
        """Write summary to TensorBoard.

        The variables to write are:
            * self.loss: Weighted multiplication of self.pos_loss and self.neg_loss.
            * self.pos_loss: Positive loss.
            * self.neg_loss: Negative loss.
            * self.metrics_dict: Metrics of positive samples.
            * self.neg_metrics_dict: Metrics of negative samples.

        Args:
            prefix: str, Tensorboard tag prefix.
            global_step: int, global step.

        Returns:
            None.
        """
        self.writer.add_scalar(f'{prefix}_loss', self.loss, global_step=global_step)
        self.writer.add_scalar(f'{prefix}_positive_loss', self.pos_loss, global_step=global_step)
        self.writer.add_scalar(f'{prefix}_negative_loss', self.neg_loss, global_step=global_step)
        for name, scalar in self.metrics_dict.items():
            self.writer.add_scalar(f'{prefix}_positive_{name}', scalar, global_step=global_step)
        for name, scalar in self.neg_metrics_dict.items():
            self.writer.add_scalar(f'{prefix}_negative_{name}', scalar, global_step=global_step)

    def to_device(self, tensor_dict):
        """Move tensor_dict to device.

        Besides, because we set batch_size=1 in data_loader, the size of tensor in the dict is
        [1, batch_size, max_seq_length]. Therefore, the size of tensor should be converted to
        [batch_size, max_seq_length].

        Args:
            tensor_dict: dict, containing PyTorch Tensors.

        Returns:
            tensor_dict.
        """
        for k, t in tensor_dict.items():
            if torch.is_tensor(t):
                tensor_dict[k] = t.squeeze(0).to(self.device)
            elif isinstance(t, dict):
                for t_k, t_t in t.items():
                    tensor_dict[k][t_k] = t_t.squeeze(0).to(self.device)
            else:
                raise NotImplementedError(f'Type {type(k)} is not implemented.')
        return tensor_dict

    def train(self):
        self.train_global_step = 0

        for epoch_i in range(self.current_epoch, self.config.epoch_num + 1):
            # Set the module in evaluation mode.
            self.model.train()
            desc = f'Epoch {epoch_i}'
            # Init for computing epoch-level metrics.
            self.epoch_loss = defaultdict(list)
            self.epoch_pos_loss = defaultdict(list)
            self.epoch_neg_loss = defaultdict(list)
            self.epoch_total_loss = []
            self.epoch_pos_pred = defaultdict(list)
            self.epoch_neg_pred = defaultdict(list)
            self.epoch_pos_label = defaultdict(list)
            self.epoch_neg_label = defaultdict(list)
            # Start training.
            tqdm_train_data_loader = tqdm(self.train_data_loader, ncols=120, desc=desc)
            for batch_i, (fea2vec, label2vec, pn2vec, sess_length) in enumerate(tqdm_train_data_loader):
                # Reset gradient.
                self.optimizer.zero_grad()
                # Convert fea2vec, label2vec and pn2vec to self.device.
                fea2vec = self.to_device(fea2vec)
                label2vec = self.to_device(label2vec)
                pn2vec = self.to_device(pn2vec)
                sess_length = sess_length.squeeze(0).to(self.device)
                # Get model output dict.
                model_outputs = self.model(fea2vec, sess_length)
                # Update train global step.
                self.train_global_step += 1
                # Get loss of each task.
                batch_loss_dict = {}
                for label_name, output_config in self.config.output_config_dict.items():
                    label_type = self.config.label_config_dict[label_name]['type']
                    # Prefix used for TensorBoard.
                    tb_prefix = f'train_batch/{label_name}'
                    if label_type == 'label':
                        # RL modified
                        self.rl_loss, self.pos_loss, self.neg_loss, _, _ = self.__process_single_label(
                            model_outputs[label_name],
                            label2vec[label_name],
                            pn2vec[label_name],
                            output_config['loss'],
                            output_config['neg_loss'],
                            label_name,
                            sess_length)
                        if self.RS == 'Supervised':
                            self.loss = self.pos_loss + self.neg_loss_weight[label_name] * self.neg_loss
                        elif self.RS == 'Reinforcement':
                            self.loss = self.rl_loss
                        else:
                            raise Exception('unknown RS:' + self.RS)
                    elif label_type == 'multi_label':
                        self.loss, _, _ = self.__process_multi_label(model_outputs[label_name],
                                                                     label2vec[label_name],
                                                                     output_config['loss'],
                                                                     label_name)
                        self.pos_loss, self.neg_loss = self.loss, 0.
                    else:
                        # TODO: computing loss when label_type = 'sim'.
                        raise NotImplementedError(f'The loss function for {label_type} is not implemented.')
                    self.epoch_loss[label_name].append(self.loss.item())
                    batch_loss_dict[label_name] = self.label_loss_weight[label_name] * self.loss
                    # Compute epoch-level metrics and write them to TensorBoard.
                    self.write_summary(tb_prefix, global_step=self.train_global_step)
                    # Print log.
                    self.print_log(label_name, self.train_global_step)
                # Compute batch total loss.
                total_batch_loss = torch.stack(tuple(batch_loss_dict.values()))
                total_batch_loss = torch.sum(total_batch_loss)
                self.epoch_total_loss.append(total_batch_loss.item())
                # Back propagation.
                total_batch_loss.backward()
                # Gradient clip.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                # Run optimizer.
                self.optimizer.step()
            # Compute epoch-level metrics and write them to TensorBoard.
            self.__process_epoch_metrics(tb_predix='train_epoch')
            # Evaluate and save model.
            if self.current_epoch % self.config.valid_interval == 0:
                self.save_model(self.current_epoch)
                self.evaluate()
            # Update current epoch
            self.current_epoch = epoch_i + 1
        # Finally evaluate and save model.
        if self.current_epoch % self.config.valid_interval != 0:
            self.save_model(self.current_epoch)
            self.evaluate()

    def evaluate(self):
        # Set the module in evaluation mode.
        self.model.eval()
        # Init for computing epoch-level metrics.
        self.epoch_loss = defaultdict(list)
        self.epoch_pos_loss = defaultdict(list)
        self.epoch_neg_loss = defaultdict(list)
        self.epoch_total_loss = []
        self.epoch_pos_pred = defaultdict(list)
        self.epoch_neg_pred = defaultdict(list)
        self.epoch_pos_label = defaultdict(list)
        self.epoch_neg_label = defaultdict(list)

        out_file = None
        if self.config.test_out_file != None:
            out_file = open(self.config.test_out_file, 'w')
        # Start evaluating.
        desc = f'Evaluating epoch {self.current_epoch}'
        tqdm_valid_data_loader = tqdm(self.valid_data_loader, ncols=120, desc=desc)
        for batch_i, (fea2vec, label2vec, pn2vec, sess_length, lines) in enumerate(tqdm_valid_data_loader):
            # Convert fea2vec, label2vec and pn2vec to self.device.
            fea2vec = self.to_device(fea2vec)
            label2vec = self.to_device(label2vec)
            pn2vec = self.to_device(pn2vec)
            sess_length = sess_length.squeeze(0).to(self.device)
            # Get model output dict.
            with torch.no_grad():
                model_outputs = self.model(fea2vec, sess_length)
            # Get loss of each task.
            batch_loss_dict = {}

            result = []
            for label_name, output_config in self.config.output_config_dict.items():
                label_type = self.config.label_config_dict[label_name]['type']
                if label_type == 'label':
                    rl_loss, pos_loss, neg_loss, _, _ = self.__process_single_label(model_outputs[label_name],
                                                                                    label2vec[label_name],
                                                                                    pn2vec[label_name],
                                                                                    output_config['loss'],
                                                                                    output_config['neg_loss'],
                                                                                    label_name,
                                                                                    sess_length)
                    if self.RS == 'Supervised':
                        loss = pos_loss + self.neg_loss_weight[label_name] * neg_loss
                    elif self.RS == 'Reinforcement':
                        loss = rl_loss
                    else:
                        raise Exception('unknown RS:' + self.RS)
                elif label_type == 'multi_label':
                    loss, _, _ = self.__process_multi_label(model_outputs[label_name],
                                                            label2vec[label_name],
                                                            output_config['loss'],
                                                            label_name)
                else:
                    # TODO: computing loss when label_type = 'sim'.
                    raise NotImplementedError(f'The loss function for {label_type} is not implemented.')
                self.epoch_loss[label_name].append(loss.item())
                batch_loss_dict[label_name] = self.label_loss_weight[label_name] * loss

            if out_file is not None:
                start_index = 0
                for i in range(len(lines)):
                    end_index = start_index + sess_length[i]
                    out_file.write(lines[i][0].rstrip())
                    for lable_name, top_probs, top_indices in result:
                        out_file.write('\t{}\t{}'.format(self.config.multi_turn_separator.join(
                            [str(self.feature_index.label_id2value[label_name][x.item()]) for x in
                             top_indices[start_index:end_index, 0]]),
                            self.config.multi_turn_separator.join([str(x.item()) for x in top_probs[start_index:end_index, 0]])))
                        start_index = end_index
                        out_file.write('\n')

            # Compute batch total loss.
            total_batch_loss = torch.stack(tuple(batch_loss_dict.values()))
            total_batch_loss = torch.sum(total_batch_loss)
            self.epoch_total_loss.append(total_batch_loss.item())

        if out_file:
            out_file.close()
        # Compute epoch-level metrics and write them to TensorBoard.
        logger.info('=========================Evaluation Results=========================')
        self.__process_epoch_metrics(tb_predix='valid_epoch')
        logger.info('=' * 68)

    def __process_single_label(self, model_outputs, labels, pns, pos_loss_type, neg_loss_type, label_name, sess_length):
        """Process single label model outputs.

        The functions of this method are:
            * Computing positive loss and negative loss.
            * Computing positive metrics and negative metrics.
            * Computing top-k probabilities and top-k indices.

        Args:
            model_outputs: Tensor, model outputs.
            labels: Tensor, labels corresponding to model outputs.
            pns: Tensor, tags (0 or 1) about whether positive sample or negative sample.
            pos_loss_type: str, loss type of positive samples.
            neg_loss_type: str, loss type of negative samples.
            label_name: str, label name.

        Returns:
            pos_loss: Scalar Tensor, positive loss.
            neg_loss: Scalar Tensor, negative loss.
            top_probs: Tensor, top-k probabilities.
            top_indices: Tensor, top-k indices.
        """
        pos_mask = torch.eq(pns, 1) & torch.ne(labels, -1)
        neg_mask = torch.eq(pns, 0) & torch.ne(labels, -1)

        pos_outputs = model_outputs[pos_mask, :]
        neg_outputs = model_outputs[neg_mask, :]
        pos_labels = labels[pos_mask]
        neg_labels = labels[neg_mask]

        pos_loss = single_label_pos_loss(pos_outputs, pos_labels, pos_loss_type)
        neg_loss = single_label_neg_loss(neg_outputs, neg_labels, neg_loss_type)

        if self.RS == 'Supervised':
            rl_loss = None
        elif self.RS == 'Reinforcement':
            rl_loss = rl_single_label_loss(model_outputs, labels, pos_loss_type, sess_length, pns)

        logits = F.softmax(model_outputs, dim=-1)

        if pos_outputs.shape[0] != 0:
            pos_predictions = torch.argmax(pos_outputs, dim=-1)
        else:
            pos_predictions = torch.tensor([], dtype=torch.long, device=self.device)

        if neg_outputs.shape[0] != 0:
            neg_predictions = torch.argmax(neg_outputs, dim=-1)
        else:
            neg_predictions = torch.tensor([], dtype=torch.long, device=self.device)

        self.metrics_dict = calc_metrics(pos_predictions, pos_labels)
        self.neg_metrics_dict = calc_metrics(neg_predictions, neg_labels)

        self.epoch_pos_loss[label_name].append(pos_loss.item())
        self.epoch_neg_loss[label_name].append(neg_loss.item())
        self.epoch_pos_pred[label_name].append(pos_predictions)
        self.epoch_neg_pred[label_name].append(neg_predictions)
        self.epoch_pos_label[label_name].append(pos_labels)
        self.epoch_neg_label[label_name].append(neg_labels)

        top_probs, top_indices = torch.topk(logits, min(self.config.topk, logits.shape[-1]))

        return rl_loss, pos_loss, neg_loss, top_probs, top_indices

    def __process_multi_label(self, model_outputs, labels, loss_type, label_name):
        """Process multi-label model outputs.

        The functions of this method are:
            * Computing multi-label loss.
            * Computing multi-label metrics.
            * Computing top-k probabilities and top-k indices.

        Args:
            model_outputs: Tensor, model outputs.
            labels: Tensor, labels corresponding to model outputs.
            loss_type: str, multi-label loss type.
            label_name:str, label name.

        Returns:
            loss: Scalar Tensor, multi-label loss.
            top_probs: Tensor, top-k probabilities.
            top_indices: Tensor, top-k indices.
        """
        loss = multi_label_loss(model_outputs, labels, loss_type)

        logits = F.sigmoid(model_outputs)
        predictions = torch.ge(logits, self.config.multi_label_threshold).float()
        self.metrics_dict = calc_multi_label_metrics(predictions, labels)
        self.neg_metrics_dict = calc_multi_label_metrics(torch.tensor([], device=self.device),
                                                         torch.tensor([], device=self.device))

        self.epoch_pos_loss[label_name].append(loss.item())
        self.epoch_pos_pred[label_name].append(predictions)
        self.epoch_pos_label[label_name].append(labels)

        top_probs, top_indices = torch.topk(logits, self.config.topk)

        return loss, top_probs, top_indices

    def __process_epoch_metrics(self, tb_predix):
        """Computing epoch-level metrics and write it to log and TensorBoard.

        The variables to compute are:
            * self.loss: Weighted multiplication of self.pos_loss and self.neg_loss.
            * self.pos_loss: Average batch positive loss in the epoch.
            * self.neg_loss: Average batch negative loss in the epoch.
            * self.metrics_dict: Metrics of positive samples.
            * self.neg_metrics_dict: Metrics of negative samples.

        Args:
            tb_predix: str, TensorBoard tag prefix.

        Returns:
            None.
        """
        for label_name, label_config in self.config.label_config_dict.items():
            label_type = label_config['type']
            if label_type == 'label':
                # Compute average batch loss.
                self.loss = np.mean(self.epoch_loss[label_name])
                self.pos_loss = np.mean(self.epoch_pos_loss[label_name])
                self.neg_loss = np.mean(self.epoch_neg_loss[label_name])
                # Concatenate predictions and labels in the epoch.
                if len(self.epoch_pos_label[label_name]) != 0:
                    pos_predictions = torch.cat(self.epoch_pos_pred[label_name])
                    pos_labels = torch.cat(self.epoch_pos_label[label_name])
                else:
                    pos_predictions, pos_labels = np.array([]), np.array([])
                if len(self.epoch_neg_label[label_name]) != 0:
                    neg_predictions = torch.cat(self.epoch_neg_pred[label_name])
                    neg_labels = torch.cat(self.epoch_neg_label[label_name])
                else:
                    neg_predictions, neg_labels = np.array([]), np.array([])
                # Compute metrics.
                self.metrics_dict = calc_metrics(pos_predictions, pos_labels)
                self.neg_metrics_dict = calc_metrics(neg_predictions, neg_labels)
            elif label_type == 'multi_label':
                # Compute average batch loss.
                self.loss = np.mean(self.epoch_loss[label_name])
                self.pos_loss = np.mean(self.epoch_pos_loss[label_name])
                self.neg_loss = 0.
                # Concatenate predictions and labels in the epoch.
                pos_predictions = torch.cat(self.epoch_pos_pred[label_name], dim=0)
                pos_labels = torch.cat(self.epoch_pos_label[label_name], dim=0)
                # Compute metrics.
                self.metrics_dict = calc_multi_label_metrics(pos_predictions, pos_labels)
                self.neg_metrics_dict = calc_multi_label_metrics(torch.tensor([], device=self.device),
                                                                 torch.tensor([], device=self.device))
            else:
                raise NotImplementedError(f'The loss function for {label_type} is not implemented.')
            # Write variables to TensorBoard and log.
            if self.config.action == 'train':
                self.write_summary(prefix=f'{tb_predix}/{label_name}', global_step=self.current_epoch)
            self.print_log(label_name)
        # Compute average total loss in the epoch.
        total_loss = np.mean(self.epoch_total_loss)
        if self.config.action == 'train':
            self.writer.add_scalar(f'{tb_predix}/avg_total_loss', total_loss, global_step=self.current_epoch)
        logger.info(f'Epoch average loss: {total_loss:.2f}.')
