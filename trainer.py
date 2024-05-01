import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import transformers
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 画图包

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

'''Custom package'''
from model import get_optimizer
from tools.rdrop import compute_kl_loss

'''trainer'''
class Trainer(nn.Module):
    def __init__(self, args, model, tokenizer, adv=None, ema=None):
        super(Trainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.adv = adv
        self.ema = ema
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
        self.optimizer = get_optimizer(self.model, self.args)
        if self.args.lr_scheduler_type == 'cosineAnnealing':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.T_0, 
                T_mult=self.args.T_mult)
        elif self.args.lr_scheduler_type == 'cosine':
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                int(self.all_iter * self.args.warmup_proportion),
                self.all_iter)
        elif self.args.lr_scheduler_type == 'linear':
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                int(self.all_iter * self.args.warmup_proportion),
                self.all_iter)
        else:
            self.scheduler = None

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
        print("***** Running training *****")
        print(f"  Num examples = {len(train_loader.dataset)}")
        print(f"  Num Epochs = {self.args.epochs}")
        print(f"  Instantaneous batch size per device = {self.args.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.all_iter}\n")

        self.create_optimizer_scheduler()

        if self.args.resume:  # keep training or not
            model_fp = os.path.join(self.args.resPath, "last.tar")
            print("load checkpoint: " + model_fp)
            checkpoint = torch.load(model_fp)
            self.model.load_state_dict(checkpoint['net'])

        self.model.train()
        Max_f1 = 0
        for epoch in range(self.args.epochs):
            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = self.prepare_pairwise_input(batch)
                losses = self.train_step(input_ids, attention_mask, labels, batch['index'])
                
                if self.gstep > self.args.max_iter:
                    break
                self.save_model(epoch, best_dev=False)
                self.gstep += 1
                
                if (self.gstep%self.args.logging_step==0) or (self.gstep==self.all_iter) or (self.gstep==self.args.max_iter):
                    metrics, wrong_predicts = self.eval(dev_loader)
                    if metrics['F1'] > Max_f1:
                        Max_f1 = metrics['F1']
                        self.save_model(epoch, best_dev=True)
                        with open(f"{self.args.resPath}/wrong_predict_best.txt", 'w') as f:
                            f.write('\n'.join(wrong_predicts))

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
        sts_metrics, wrong_predicts = self.eval(dev_loader)
        losses.update(sts_metrics)
        return sts_metrics

    def train_step(self, input_ids, attention_mask, labels, index=None):
        
        p = self.model(input_ids, attention_mask)
        if self.args.use_rdrop:
            p1 = self.model(input_ids, attention_mask)
            ce_loss = 0.5 * (self.loss_func(p, labels) + self.loss_func(p1, labels))
            kl_loss = compute_kl_loss(p, p1)
            loss = ce_loss + self.args.alpha * kl_loss
        else:
            loss = self.loss_func(p, labels)

        losses = {}
        losses["loss"] = loss.item()

        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)

        if self.args.use_attack:
            self.adv.attack()
            p = self.model(input_ids, attention_mask)        
            if self.args.use_rdrop:
                p1 = self.model(input_ids, attention_mask)
                ce_loss = 0.5 * (self.loss_func(p, labels) + self.loss_func(p1, labels))
                kl_loss = compute_kl_loss(p, p1)
                loss_adv = ce_loss + self.args.alpha * kl_loss
            else:
                loss_adv = self.loss_func(p, labels)
            # loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.adv.restore()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update()
        
        self.optimizer.zero_grad()
        losses["lr"] = self.scheduler.get_lr()[1]
        return losses

    @torch.no_grad()
    def eval(self, data_loader):
        y_pred = []  # predict labels
        y = []  # ground truth labels
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        for step, batch in enumerate(data_loader):
            input_ids, attention_mask, labels = self.prepare_pairwise_input(batch)
            p = self.model(input_ids, attention_mask)
            preds = torch.argmax(p, dim=1)
            y_pred.extend(preds.cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
        if self.ema is not None:
            self.ema.restore()
        self.model.train()
        y_pred = np.array(y_pred)
        y = np.array(y)

        index = y != y_pred
        wrong_predicts = [
            'true_label:'+str(y[id]) + '\tpred_label:'+str(y_pred[id]) + '\t'+data_loader.dataset[id]['sent'] for id, diff in enumerate(index) if diff
        ]
        report = metrics.classification_report(y, y_pred, digits=4, output_dict=True, zero_division=np.nan) # output_dict=True
        Roc_auc = metrics.roc_auc_score(y, y_pred)

        return {
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1': report['macro avg']['f1-score'],
            'Roc_auc': Roc_auc
        }, wrong_predicts

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

        index = y != y_pred
        wrong_predicts = [
            'true_label:'+str(y[id]) + '\tpred_label:'+str(y_pred[id]) + '\t'+data_loader.dataset[id]['sent'] for id, diff in enumerate(index) if diff
        ]
        report = metrics.classification_report(y, y_pred, digits=4, output_dict=True, zero_division=np.nan) # output_dict=True
        Roc_auc = metrics.roc_auc_score(y, y_pred)

        return {
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1': report['macro avg']['f1-score'],
            'Roc_auc': Roc_auc
        }, wrong_predicts

    def save_model(self, epoch, best_dev=False):     
        model1 = copy.deepcopy(self.model)
        model1 = model1.to('cpu')
        state = {'net': model1.state_dict(), 'epoch': epoch}
        if best_dev:
            # save the ckpt according to the performance on dev set   
            torch.save(state, os.path.join(self.args.resPath, 'best.tar'))
        else:
            # save the ckpt per epoch
            torch.save(state, os.path.join(self.args.resPath, f'last.tar'))
            # self.model.save_pretrained(os.path.join(self.args.resPath, f'epoch_{epoch}'))
            # self.tokenizer.save_pretrained(os.path.join(self.args.resPath, f'epoch_{epoch}'))

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

