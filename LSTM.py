import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import data

# 下载停用词和分词器
nltk.download('punkt')
nltk.download('stopwords')


class DataPreprocessing:
    def __init__(self, csv_path):
        self.dataset = pd.read_csv(csv_path, encoding='Windows-1252')

    def clean_and_tokenize(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.word_tokenize(text)
            return tokens
        else:
            return []

    def preprocess(self):
        self.dataset['text'] = self.dataset['text'].apply(self.clean_and_tokenize)

        stop_words = set(stopwords.words('english'))
        self.dataset['text'] = self.dataset['text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

        stemmer = PorterStemmer()
        self.dataset['text'] = self.dataset['text'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

        self.dataset.to_csv('preprocessed_dataset.csv', index=False)


class SentimentAnalysisRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SentimentAnalysisRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# 数据预处理
preprocessing = DataPreprocessing("C:/Users/29508/Desktop/test.csv")
preprocessing.preprocess()

# 使用torchtext加载数据集
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = data.TabularDataset.splits(
    path='.',  # 保存预处理数据的路径
    train='preprocessed_dataset.csv',
    test='preprocessed_dataset.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)

# 构建词汇表
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

# 初始化模型、损失函数和优化器
model = SentimentAnalysisRNN(len(TEXT.vocab), 128, 256, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    for batch in data.BucketIterator(train_data, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        text, labels = batch.text, batch.label
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# 评估
model.eval()
with torch.no_grad():
    for batch in data.BucketIterator(test_data, batch_size=32, shuffle=False):
        text, labels = batch.text, batch.label
        predictions = model(text).squeeze(1)
        # 根据实际需求执行评估
