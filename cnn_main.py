import os
import sys
import json
import copy
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as util_data

import transformers

from sklearn import metrics

'''Custom package'''
from utils import seed_everything
from dataloader import get_dataloader
from model import get_bert, Network_CNN
# 画图包

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Trainer(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(Trainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.loss_func = nn.CrossEntropyLoss()

        self.gstep = 0
        self.dev_objective = -np.inf

        self.acc_list = []
        self.f1_list = []
        self.recal_list = []
        self.precision_list = []
        self.loss_list = []

    def create_optimizer_scheduler(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            600,
            self.all_iter
        )

    def get_batch_token(self, texts):
        token_feats = self.tokenizer.batch_encode_plus(
            texts, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )

        return token_feats

    def prepare_pairwise_input(self, batch, aug=None):
        texts = batch['sent']
        labels = batch['label']
        feat = self.get_batch_token(texts)

        input_ids = feat['input_ids']
        attention_mask = feat['attention_mask']
        return input_ids.to(self.args.device), attention_mask.to(self.args.device), labels.to(self.args.device)
        
    def train(self, train_loader, dev_loader):
        # training info
        self.all_iter = self.args.epochs * len(train_loader)
        self.all_iter = min(self.all_iter, self.args.max_iter)
        print("***** Running training *****")
        print(f"  Num examples = {len(train_loader.dataset)}")
        print(f"  Num Epochs = {self.args.epochs}")
        print(f"  Instantaneous batch size per device = {self.args.batch_size}")
        print(f"  Max train batch size (w. parallel, distributed & accumulation) = {self.all_iter}\n")

        self.create_optimizer_scheduler()

        self.model.train()
        Max_f1 = 0
        for epoch in range(self.args.epochs):
            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = self.prepare_pairwise_input(batch)
                losses = self.train_step(input_ids, attention_mask, labels, batch['index'])
                
                if self.gstep > self.args.max_iter:
                    break
                self.gstep += 1
                
                if (self.gstep%self.args.logging_step==0) or (self.gstep==self.all_iter) or (self.gstep==self.args.max_iter):
                    metrics = self.eval(dev_loader)
                    if metrics['F1'] > Max_f1:
                        Max_f1 = metrics['F1']

                    losses.update(metrics)    
                    print(f"[Step:{self.gstep:4}/{self.all_iter:4}]\n",
                        f"learning rate: {losses['lr']:2.4}", 
                        f"loss: {losses['loss']:2.4}\n", 
                        f"Accuracy: {losses['Accuracy']:2.2%}", 
                        f"Precision: {losses['Precision']:2.2%}", 
                        f"Recall: {losses['Recall']:2.2%}", 
                        f"F1: {losses['F1']:2.2%}", 
                        f"Roc_auc: {losses['Roc_auc']:2.2%}\n",    
                    )
                    self.acc_list.append(losses['Accuracy'])
                    self.f1_list.append(losses['F1'])
                    self.recal_list.append(losses['Recall'])
                    self.precision_list.append(losses['Precision'])
                    self.loss_list.append(losses['loss'])

        self.plot_()
        sts_metrics = self.eval(dev_loader)
        losses.update(sts_metrics)
        return sts_metrics

    def train_step(self, input_ids, attention_mask, labels, index=None):
        
        p = self.model(input_ids, attention_mask)
        loss = self.loss_func(p, labels)

        losses = {}
        losses["loss"] = loss.item()

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        losses["lr"] = self.scheduler.get_lr()[0]
        return losses

    @torch.no_grad()
    def eval(self, data_loader):
        y_pred = []  # predict labels
        y = []  # ground truth labels
        self.model.eval()
        for step, batch in enumerate(data_loader):
            input_ids, attention_mask, labels = self.prepare_pairwise_input(batch)
            p = self.model(input_ids, attention_mask)
            preds = torch.argmax(p, dim=1)
            y_pred.extend(preds.cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
        self.model.train()
        y_pred = np.array(y_pred)
        y = np.array(y)
        
        report = metrics.classification_report(y, y_pred, digits=4, output_dict=True, zero_division=np.nan) # output_dict=True
        Roc_auc = metrics.roc_auc_score(y, y_pred)

        return {
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1': report['macro avg']['f1-score'],
            'Roc_auc': Roc_auc
        }

    @torch.no_grad()
    def test(self, data_loader):
        y_pred = []  # predict labels
        y = []  # ground truth labels
        self.model.eval()
        for step, batch in enumerate(data_loader):
            input_ids, attention_mask, labels = self.prepare_pairwise_input(batch)
            p = self.model(input_ids, attention_mask)
            preds = torch.argmax(p, dim=1)
            y_pred.extend(preds.cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
        y_pred = np.array(y_pred)
        y = np.array(y)
        
        report = metrics.classification_report(y, y_pred, digits=4, output_dict=True, zero_division=np.nan) # output_dict=True
        Roc_auc = metrics.roc_auc_score(y, y_pred)

        return {
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1': report['macro avg']['f1-score'],
            'Roc_auc': Roc_auc
        }

    def plot_(self):  # 画图函数

        def sub_plot(X, Y, name):
            # loss 随着训练的变化
            plt.figure(figsize=(6, 4))
            plt.plot(X, Y, label=name, color='b', linestyle='-.', linewidth=1.1)
            plt.xlabel('Step', size=12)
            plt.ylabel(name, size=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', alpha=0.1) #alpha 网格颜色深度
            plt.legend(loc='upper left')  # 图例大小
            plt.savefig(self.args.resPath+name+'.png', dpi=400)
            plt.cla()
        
        X = [(i+1)*20 for i in range(len(self.loss_list))]
        # loss 随着训练的变化
        sub_plot(X, self.loss_list, 'loss')

        # accuracy 随着训练的变化
        sub_plot(X, self.acc_list, 'Accuracy')

        # Recall 随着训练的变化
        sub_plot(X, self.recal_list, 'Recall')

        # precision 随着训练的变化
        sub_plot(X, self.precision_list, 'Precision')

        # precision 随着训练的变化
        sub_plot(X, self.f1_list, 'F1')


def run(train_loader, dev_loader, test_loader, args):
    # load model
    bert, tokenizer = get_bert(args.pretrained_model_path)
    model = Network_CNN(bert, args)
    # model = nn.DataParallel(model)
    model.to(args.device)

    # training
    trainer = Trainer(args, model, tokenizer)
    sts_metrics = trainer.train(train_loader, dev_loader)

    # testing
    print("\n\n***** Running testing *****")
    metrics = trainer.test(test_loader)
    print(f"最终在测试集上的结果：",
        f"Accuracy: {metrics['Accuracy']:2.2%}", 
        f"Precision: {metrics['Precision']:2.2%}", 
        f"Recall: {metrics['Recall']:2.2%}", 
        f"F1: {metrics['F1']:2.2%}", 
        f"Roc_auc: {metrics['Roc_auc']:2.2%}",    
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",  default='CLS', type=str, help="")
    parser.add_argument("--seed",       default=2024, type=int, help="随机种子的设置")
    parser.add_argument("--workers",    default=4, type=int, help="")

    '''dataset'''
    parser.add_argument("--data_dir", default="003/data/sst2_shuffled.tsv", type=str, help="")
    parser.add_argument("--train_ratio",    default=0.8, type=float, help="训练集占比")
    parser.add_argument("--class_num",  default=2, type=int, help="")
    parser.add_argument("--max_length", default=64,type=int, help="文本的最大长度")

    '''model 参数'''
    parser.add_argument("--pretrained_model_path",  default='model/google-bert/bert-base-uncased',   type=str, help="使用的预训练词向量模型，不参与训练")
    parser.add_argument("--kernel_size",default=[3, 4, 5], nargs='+', type=int, help="")
    parser.add_argument("--num_kernels",default=128, type=int, help="")
    parser.add_argument("--stride",     default=1, type=int, help="")

    '''training'''
    parser.add_argument("--batch_size", default=10,      type=int, help="训练批次的大小")
    parser.add_argument("--epochs",     default=4,    type=int, help="训练的轮次")
    parser.add_argument("--max_iter",   default=500,   type=int, help="训练的step")
    parser.add_argument("--lr",    default=0.0001, type=float, help="学习率")    
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="")
    parser.add_argument("--resume",     default=0, type=int, help="是否接着上次进行训练")
    parser.add_argument("--logging_step",     default=20, type=int, help="")
    parser.add_argument("--check_point_path", default="./save_CNN/", type=str, help="断点的存储地方")

    args = parser.parse_args()
    return args



def main():
    # 获得代码运行的相关参数
    args = get_args()

    args.resPath = args.check_point_path
    if not os.path.exists(args.resPath):
        os.makedirs(args.resPath)
    # 固定随机种子
    seed_everything(args.seed)
    # 使用训练模型使用  CPU   or   GPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    print(args.device)
    torch.backends.cudnn.enabled = False

    # load dataset
    train_loader, test_loader = get_dataloader(args.data_dir, args.batch_size, args.train_ratio, args.seed, args.workers)
    
    # train and test
    run(train_loader, test_loader, test_loader, args)


if __name__ == '__main__':
    main()