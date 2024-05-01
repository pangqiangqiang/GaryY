import argparse


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

    '''model'''
    parser.add_argument("--bert_name_path",  default='model/google-bert/bert-base-uncased',   type=str, help="使用该种bert的哪一个")
    parser.add_argument("--kernel_size",default=[3, 4, 5], nargs='+', type=int, help="")
    parser.add_argument("--num_kernels",default=128, type=int, help="")
    parser.add_argument("--stride",     default=1, type=int, help="")

    '''training'''
    parser.add_argument("--batch_size", default=32,      type=int, help="训练批次的大小")
    parser.add_argument("--epochs",     default=40,    type=int, help="训练的轮次")
    parser.add_argument("--max_iter",   default=6000,   type=int, help="训练的step")
    parser.add_argument("--bert_lr",    default=0.00001, type=float, help="预训练模型的学习率")
    parser.add_argument("--head_lr",    default=0.00005, type=float, help="MLP的学习率")
    parser.add_argument("--resume",     default=0, type=int, help="是否接着上次进行训练")
    parser.add_argument("--logging_step",     default=20, type=int, help="")
    parser.add_argument("--check_point_path", default="./save/", type=str, help="断点的存储地方")


    '''fine-tuning tracks'''

    # noisy tuning
    parser.add_argument("--use_noise", default=0, type=int, help="是否给预训练模型加噪声")
    parser.add_argument("--noise_lambda", default=0.15, type=float, help="增加噪声时的超参数")

    # attack training
    parser.add_argument("--use_attack", default=0, type=int, help="是否使用对抗训练")
    parser.add_argument("--epsilon",  default=0.0001, type=float, help="")
    parser.add_argument("--attack_type", default='FGM', type=str, help="使用哪一种对抗训练", choices=['FGM', 'PGD', 'FreeLB'])

    # EMA  https://zhuanlan.zhihu.com/p/68748778
    parser.add_argument("--use_ema", default=0, type=int, help="是否使用指数平均移动")

    # R-Drop
    parser.add_argument("--use_rdrop", default=0, type=int, help="是否使用R-Drop")
    parser.add_argument("--alpha",  default=0.7, type=float, help="")

    # Mixout
    parser.add_argument("--mixout", default=0, type=float, help="是否使用Mixout")

    # warmup & lr scheduler 学习率衰减策略
    parser.add_argument("--lr_scheduler_type", default='cosineAnnealing', type=str, help="学习率的变化", choices=['linear', 'cosine', 'cosineAnnealing'])
    parser.add_argument("--warmup_proportion", default=0.05, type=float, help="warmup 的比例。0表示不使用")
    parser.add_argument("--T_0",    default=5, type=int,     help="")
    parser.add_argument("--T_mult", default=1, type=int, help="")

    # optimizer choice
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="")
    parser.add_argument("--correct_bias", default=1, type=int, help="偏差矫正")

    # bert 一些层数的变化
    parser.add_argument("--freeze_layer_count", default=0, type=int, help="冻结bert的前几层. 0 not freeze anything, -1 freeze embedding layer only")
    parser.add_argument("--reinit_layers", default=0, type=int, help="初始化EBRT后面几层的参数")
    parser.add_argument("--reinit_pooler", default=0, type=int, help="初始化EBRT后面的pooler")


    args = parser.parse_args()
    return args