import os
import sys
import json
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import transformers
import torch.utils.data as util_data

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

'''Custom package'''
from init_args import get_args
from utils import setup_path, seed_everything
from dataloader import get_dataloader
from model import Network, get_bert, get_optimizer
from trainer import Trainer
from tools.mixout import MixLinear
from tools.attack_training import FGM, FreeLB, PGD
from tools.ema import EMA


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(train_loader, dev_loader, test_loader, args, fold_n=0):
    # load model
    bert, tokenizer = get_bert(args.bert_name_path)
    model = Network(bert, args)
    # model = nn.DataParallel(model)
    model.to(args.device)

    # noisy tuning
    if args.use_noise:  
        model.to('cpu')
        for name, para in model.bert.named_parameters():
            model.bert.state_dict()[name][:] += (torch.rand(para.size()) - 0.5)*args.noise_lambda*torch.std(para)
        model.to(args.device)

    if args.mixout > 0:
        print('Initializing Mixout Regularization')
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout
                    ).to(args.device)
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
        print(f"use Mixout success...")

    # reinit pooler-layer
    if args.reinit_pooler:
        print(f"reinit pooler layer of {args.bert_name}")
        encoder_temp = model.bert  # getattr(model.bert, args.bert_name)
        encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
        encoder_temp.pooler.dense.bias.data.zero_()
        for p in encoder_temp.pooler.parameters():
            p.requires_grad = True

    # reinit encoder layers
    if args.reinit_layers > 0:
        # assert args.reinit_pooler
        print(f"reinit  layers count of {str(args.reinit_layers)}")
        encoder_temp = model.bert    
        for layer in encoder_temp.encoder.layer[-args.reinit_layers:]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    # freeze some layers
    if args.freeze_layer_count:
        print(f"frozen layers count of {str(args.freeze_layer_count)}")
        # We freeze here the embeddings of the model
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        if args.freeze_layer_count != -1:
            for layer in model.bert.encoder.layer[:args.freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    if args.use_attack:
        if args.attack_type == 'FGM':
            adv = FGM(model=model, epsilon=args.epsilon, emb_name='word_embeddings')
        elif args.attack_type == 'PGD':
            pass
        elif args.attack_type == 'FreeLB':
            pass
        print(f"use {args.attack_type} success...")
    else:
        adv = None

    if args.use_ema:
        ema = EMA(model, 0.999)
        ema.register()
        print("use ema success...")
    else:
        ema = None

    # training
    trainer = Trainer(args, model, tokenizer, adv, ema)
    sts_metrics = trainer.train(train_loader, dev_loader)

    os.rename(os.path.join(args.resPath, f'last.tar'), os.path.join(args.resPath, f'{fold_n}_last.tar'))
    os.rename(os.path.join(args.resPath, f'best.tar'), os.path.join(args.resPath, f'{fold_n}_best.tar'))
    os.rename(os.path.join(args.resPath, f'wrong_predict_best.txt'), os.path.join(args.resPath, f'{fold_n}_wrong_predict_best.txt'))

    with open(args.resPath + f'{fold_n}_dev_res.json', 'w') as f:
        f.write(json.dumps(sts_metrics, ensure_ascii=False))

    # testing
    print("\n\n***** Running testing *****")
    metrics, wrong_predicts = trainer.test(test_loader)
    print(f"最终在测试集上的结果：",
        f"Accuracy: {metrics['Accuracy']:2.2%}", 
        f"Precision: {metrics['Precision']:2.2%}", 
        f"Recall: {metrics['Recall']:2.2%}", 
        f"F1: {metrics['F1']:2.2%}", 
        f"Roc_auc: {metrics['Roc_auc']:2.2%}",    
    )

def main():
    # 获得代码运行的相关参数
    args = get_args()

    args.resPath = setup_path(args)
    if not os.path.exists(args.resPath):
        os.makedirs(args.resPath)
    # 固定随机种子
    seed_everything(args.seed)
    # 使用训练模型使用  CPU   or   GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)

    # load dataset
    train_loader, test_loader = get_dataloader(args.data_dir, args.batch_size, args.train_ratio, args.seed, args.workers)
    
    # train and test
    run(train_loader, test_loader, test_loader, args)


if __name__ == '__main__':
    main()