import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from box import Box
from src.preprocess import DataProcessor
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score, accuracy_score
from src.common import set_seed
from src.model import SITCL

class Main:
    def __init__(self, args):
        config = Box(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        for k, v in vars(args).items():
            if v is not None:
                setattr(config, k, v)
        self.config = config
        self.formatted_time = config.time
        self.log_dir = f"./result/{self.config.model_type}_{self.formatted_time}/log/"
        self.pred_dir = f"./result/{self.config.model_type}_{self.formatted_time}/pred/"
        self.save_dir = f"./result/{self.config.model_type}_{self.formatted_time}/save/"
        self.pred_file = self.pred_dir + f'{self.config.seed}.csv'
        for d in [self.log_dir, self.pred_dir, self.save_dir]:
            os.makedirs(d, exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir + f'{self.config.seed}.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        set_seed(self.config.seed)
        self.config.device = torch.device('cuda:{}'.format(self.config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.best_epoch = 0
        self.best_dev_macro_f1 = 0.0
        self._current_stage = None
        self.training_stage = 'all'

    def _set_requires_grad_for_stage(self, stage):
        for name, param in self.model.named_parameters():
            if stage == 'a':
                param.requires_grad = 'hypergraph_encoder' in name
            elif stage == 'b':
                param.requires_grad = (
                    'hypergraph_encoder' in name
                    or 'topology_encoder' in name
                    or 'fusion_gate' in name
                    or name.startswith('fc.')
                )
            else:
                param.requires_grad = True

    def _apply_training_stage(self, epoch):
        if not bool(getattr(self.config, 'staged_training', 0)):
            if self._current_stage != 'all':
                self._current_stage = 'all'
                self.training_stage = 'all'
                self._set_requires_grad_for_stage('all')
                self.load_param()
            return

        stage_a = int(getattr(self.config, 'stage_a_epochs', 2))
        stage_b = int(getattr(self.config, 'stage_b_epochs', 3))
        if epoch < stage_a:
            stage = 'a'
        elif epoch < stage_a + stage_b:
            stage = 'b'
        else:
            stage = 'c'

        if stage == self._current_stage:
            return

        self._current_stage = stage
        self.training_stage = stage
        self._set_requires_grad_for_stage(stage)
        self.load_param()
        logging.info('Switched training stage to %s at epoch %d', stage, epoch + 1)

    def train_iter(self):
        self.model.train()
        running_loss = 0.0
        for data in tqdm(self.trainLoader):
            loss, _, _ = self.model(**data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            running_loss += loss.item()
        train_loss = running_loss / len(self.trainLoader)
        return train_loss

    def evaluate_iter(self, dataLoader=None, mode=None):
        self.model.eval()
        val_loss = 0.0
        seq_preds = []
        seq_trues = []
        doc_id_lst = []
        if mode == 'dev':
            dataLoader = self.devLoader
        elif mode == 'test':
            dataLoader = self.testLoader
        with torch.no_grad():
            for data in tqdm(dataLoader):
                loss, seq_output, labels = self.model(**data)
                val_loss += loss.item()
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                labels = labels.detach().cpu().numpy()
                labels = labels.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(labels)
                doc_id_lst.extend(data['doc_id'])
        val_loss /= len(dataLoader)
        macro_f1, favor, against, neutral, f1_avg, acc = self.get_metrices(seq_trues, seq_preds)
        if mode == 'test':
            result = {'doc_id': doc_id_lst, 'true': seq_trues, 'pred': seq_preds}
            df = pd.DataFrame(result)
            df.to_csv(self.pred_file, index=False)
        return macro_f1, val_loss, favor, against, neutral, f1_avg, acc

    def train(self):
        best_dev_f1 = -1.0
        stale_epochs = 0
        patience = int(getattr(self.config, 'patience', 3))
        best_ckpt = self.save_dir + 'best_model.pth'

        for epoch in range(self.config.epoch_size):
            self._apply_training_stage(epoch)
            if hasattr(self.model, 'set_train_epoch'):
                self.model.set_train_epoch(epoch)
            train_loss = self.train_iter()
            dev_macro_f1, dev_loss, favor_f1, against_f1, neutral_f1, _, dev_acc = self.evaluate_iter(mode='dev')

            log_msg = (
                'Epoch %d, Train Loss: %.2f, Val Loss: %.2f, Val Macro F1: %.2f'
                % (epoch + 1, train_loss, dev_loss, 100 * dev_macro_f1)
            )
            logging.info(log_msg)

            if dev_macro_f1 > best_dev_f1 + 1e-4:
                best_dev_f1 = dev_macro_f1
                stale_epochs = 0
                self.best_epoch = epoch + 1
                self.best_dev_macro_f1 = best_dev_f1
                torch.save(self.model.state_dict(), best_ckpt)
                test_macro_f1, test_loss, _, _, _, _, _ = self.evaluate_iter(mode='test')
                logging.info(
                    'Test Loss: %.2f, Test Macro F1: %.2f',
                    test_loss,
                    100 * test_macro_f1,
                )
            else:
                stale_epochs += 1

            if patience > 0 and stale_epochs >= patience:
                logging.info(
                    'Early stop at epoch %d; best epoch=%d, best dev Macro F1=%.2f',
                    epoch + 1,
                    self.best_epoch,
                    100 * best_dev_f1,
                )
                break

        if os.path.isfile(best_ckpt):
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.config.device),
            )
            logging.info('Loaded best dev checkpoint from epoch %d', self.best_epoch)

        test_macro_f1, test_loss, favor_f1, against_f1, neutral_f1, _, test_acc = self.evaluate_iter(mode='test')
        logging.info(
            'Test Loss: %.2f, Test Macro F1: %.2f',
            test_loss,
            100 * test_macro_f1,
        )

    def load_param(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        stage = getattr(self, 'training_stage', 'all')

        if stage == 'c':
            bert_lr = float(getattr(self.config, 'bert_lr_stage_c', 1e-6))
            other_lr = float(getattr(self.config, 'other_lr_stage_c', 1e-6))
            hyper_lr = float(getattr(self.config, 'hypergraph_lr_stage_c', 5e-6))
        elif stage == 'b':
            bert_lr = 0.0
            other_lr = float(getattr(self.config, 'other_lr_stage_b', 2e-6))
            hyper_lr = float(getattr(self.config, 'hypergraph_lr', 1e-5))
        elif stage == 'a':
            bert_lr = 0.0
            other_lr = 0.0
            hyper_lr = float(getattr(self.config, 'hypergraph_lr', 1e-5))
        else:
            bert_lr = float(self.config.bert_lr)
            other_lr = float(self.config.other_lr)
            hyper_lr = float(getattr(self.config, 'hypergraph_lr', 1e-5))

        wd = float(self.config.weight_decay)

        def is_hyper(name):
            return 'hypergraph_encoder' in name

        def trainable(name, param):
            return param.requires_grad

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' in n and not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' in n and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' not in n and is_hyper(n) and not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'lr': hyper_lr},
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' not in n and is_hyper(n) and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': hyper_lr},
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' not in n and not is_hyper(n) and not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'lr': other_lr},
            {'params': [p for n, p in param_optimizer if trainable(n, p) and 'bert' not in n and not is_hyper(n) and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': other_lr},
        ]
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group['params']) > 0
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, eps=float(self.config.adam_epsilon))
        total_steps = self.config.epoch_size * len(self.trainLoader)
        warmup_steps = int(getattr(self.config, 'warmup_steps', 0))
        if warmup_steps <= 0:
            warmup_steps = int(
                float(getattr(self.config, 'warmup_proportion', 0.1)) * total_steps
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def get_metrices(self, trues, preds):   
        f1_macro = f1_score(y_true=trues, y_pred=preds, average='macro')
        f1_per_class = f1_score(y_true=trues, y_pred=preds, average=None)
        favor = f1_per_class[0]
        against = f1_per_class[1]
        neutral = f1_per_class[2]
        f1_avg = (favor + against) / 2
        accuracy = accuracy_score(y_true=trues, y_pred=preds)
        return f1_macro, favor, against, neutral, f1_avg, accuracy

    def forward(self):
        self.trainLoader, self.devLoader, self.testLoader = DataProcessor(self.config).get_data()
        self.model = SITCL(self.config).to(self.config.device)
        init_checkpoint = getattr(self.config, 'init_checkpoint', None)
        if init_checkpoint:
            state_dict = torch.load(init_checkpoint, map_location=self.config.device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            logging.info(
                'Loaded init checkpoint from %s (missing=%d, unexpected=%d)',
                init_checkpoint,
                len(missing),
                len(unexpected),
            )
        self.load_param()
        if bool(getattr(self.config, 'staged_training', 0)):
            self._set_requires_grad_for_stage('a')
            self._current_stage = None
            self._apply_training_stage(0)
        logging.info('Start training...')
        self.train()
        logging.info('End training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--other_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--time', type=str, default='')
    parser.add_argument('--gru_layer', type=int, default=1)
    parser.add_argument('--gru_hidden', type=int, default=768)
    parser.add_argument('--model_type', type=str, default='SITPCL')
    parser.add_argument('--plm', type=str, default='roberta')
    parser.add_argument('--bert_dir', type=str, default='./plm/chinese-roberta-wwm-ext/')
    parser.add_argument('--init_checkpoint', type=str, default=None)
    args = parser.parse_args()
    main = Main(args)
    main.forward()
