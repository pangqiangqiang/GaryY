import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig


def get_bert(bert_name_path):
    config = AutoConfig.from_pretrained(bert_name_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_name_path, use_fast=True)
    model = AutoModel.from_pretrained(bert_name_path, config=config)
    return model, tokenizer


'''model'''


class Network(nn.Module):
    def __init__(self, bert, args):
        super(Network, self).__init__()
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # cls head
        # self.cls_head = nn.Sequential(
        #     nn.SiLU(inplace=True),
        #     nn.Linear(self.hidden_dim, args.class_num),
        #     nn.Softmax(dim=1)
        # )

        self.cls_head = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, args.class_num),
            nn.Softmax(dim=1)
        )

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(
            input_ids=input_ids, attention_mask=attention_mask)
        words_emb = self.dropout(bert_output[0])
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(words_emb*attention_mask,
                                dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def forward(self, input_ids, attention_mask):
        h = self.get_mean_embeddings(
            input_ids=input_ids, attention_mask=attention_mask)
        p = self.cls_head(h)
        return p

    def forward_h(self, input_ids, attention_mask):
        h = self.get_mean_embeddings(input_ids, attention_mask)
        return h


class Network_CNN(nn.Module):
    def __init__(self, bert, args):
        super(Network_CNN, self).__init__()
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size
        # for param in self.bert.parameters():
        #     param.requires_grad = False


        self.num_kernels = args.num_kernels
        self.kernel_size = args.kernel_size
        self.stride = args.stride

        self.conv_0 = nn.Conv2d(
            1, self.num_kernels, (self.kernel_size[0], self.hidden_dim), self.stride)
        self.conv_1 = nn.Conv2d(
            1, self.num_kernels, (self.kernel_size[1], self.hidden_dim), self.stride)
        self.conv_2 = nn.Conv2d(
            1, self.num_kernels, (self.kernel_size[2], self.hidden_dim), self.stride)

        self.dropout = nn.Dropout(0.5)
        # cls head
        self.cls_head = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(len(self.kernel_size) *
                      self.num_kernels, args.class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        emb = bert_output[0]
        emb = emb.unsqueeze(dim=1)  # (batch, channel=1, seq_len, emb_size)

        # after conv: (batch, num_kernels, seq_len - kernel_size[0] + 1, 1)
        conved0 = F.relu(self.conv_0(emb).squeeze(3))
        conved1 = F.relu(self.conv_1(emb).squeeze(3))
        conved2 = F.relu(self.conv_2(emb).squeeze(3))

        # pooled: (batch, n_channel)
        pool0 = nn.MaxPool1d(conved0.shape[2], self.stride)
        pool1 = nn.MaxPool1d(conved1.shape[2], self.stride)
        pool2 = nn.MaxPool1d(conved2.shape[2], self.stride)

        pooled0 = pool0(conved0).squeeze(2)
        pooled1 = pool1(conved1).squeeze(2)
        pooled2 = pool2(conved2).squeeze(2)

        # (batch, n_chanel * num_filters)
        cat_pool = torch.cat([pooled0, pooled1, pooled2], dim=1)
        p = self.cls_head(cat_pool)
        return p



def get_optimizer(model, args):
    model_param = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    model_grouped_parameters = [
        {'params': [p for n, p in model_param
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in model_param
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.bert_lr}
    ]

    optimizer = transformers.AdamW(
        model_grouped_parameters,
        correct_bias=args.correct_bias,
        no_deprecation_warning=True
    )
    # optimizer = torch.optim.AdamW(model_grouped_parameters)
    return optimizer


def get_optimizer_old(model, args):
    bert_param = list(model.bert.named_parameters())
    head_param = list(model.cls_head.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_grouped_parameters = [
        {'params': [p for n, p in bert_param
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.bert_lr}
    ]
    head_grouped_parameters = [
        {'params': [p for n, p in head_param
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.head_lr},
        {'params': [p for n, p in head_param
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.head_lr},
    ]

    optimizer = transformers.AdamW(
        bert_grouped_parameters + head_grouped_parameters,
        correct_bias=args.correct_bias,
    )

    return optimizer
