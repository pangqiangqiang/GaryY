import os
import random
import numpy as np

import torch

def seed_everything(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_path(args):
    bert_name = args.bert_name_path.split('/')[-1]
    resPath = ''
    resPath += f'{bert_name}'
    resPath += f'.lr{args.bert_lr}'
    resPath += f'.bs{args.batch_size}'
    if args.use_noise: resPath += f'.use_noise'
    if args.use_attack: resPath += f'.{args.attack_type}{args.epsilon}'
    if args.use_ema: resPath += f'.use_ema'
    if args.use_rdrop: resPath += f'.use_rdrop{args.alpha}'
    if args.mixout: resPath += f'.mixout'
    resPath += f'.seed{args.seed}/'
    resPath = os.path.join(args.check_point_path, resPath)
    print(f'results path: {resPath}')

    return resPath


def statistics_log(tensorboard, losses=None, global_step=0):
    print("[Step:{}]-----".format(global_step))
    if losses is not None:
        for key, val in losses.items():
            try:
                tensorboard.add_scalar('train/'+key, val.item(), global_step)
            except:
                tensorboard.add_scalar('train/'+key, val, global_step)
        flag = True
        for key, val in losses.items():
            if flag:
                print(f"{key}: {val:3.4}", end='  ')
                if key == 'lr': 
                    flag = False
                    print()
            else:
                print(f"{key}: {val:2.3%}", end='  ')
        print()
