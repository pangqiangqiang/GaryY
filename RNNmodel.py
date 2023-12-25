import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data  # 使用 torchtext 的 legacy 模块
from Preprocessing import Preprocessing  # 导入你的Preprocessing类
import pandas as pd
import matplotlib.pyplot as plt

# 创建Preprocessing的实例并加载预处理数据集
preprocessing_instance = Preprocessing(file_path="C:/Users/29508/Desktop/test.csv")

# 从Preprocessing实例中访问预处理数据集
preprocessed_dataset = preprocessing_instance.dataset

# 将预处理数据集保存到CSV文件
preprocessed_dataset.to_csv("C:/Users/29508/Desktop/preprocessed_data.csv", index=False)

# 使用TorchText加载数据
TEXT = data.Field(tokenize=lambda x: x.split(), lower=True)  # 使用空格分词
LABEL = data.LabelField(dtype=torch.float)

# 定义fields映射
fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

# 将处理后的数据集传递给TorchText的数据集创建函数
train_data, test_data = data.TabularDataset.splits(
    path='.', train='preprocessed_data.csv', test='test.csv', format='csv',
    fields=fields)

# 构建词汇表
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)
# 定义模型
class SentimentAnalysisRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SentimentAnalysisRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output

# 初始化模型、损失函数和优化器
embedding_dim = 128
hidden_size = 256
output_size = 1
learning_rate = 0.01
num_epochs = 10

model = SentimentAnalysisRNN(len(TEXT.vocab), embedding_dim, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环及误差可视化
train_losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_data:
        optimizer.zero_grad()
        text, labels = batch.text, batch.label
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    train_losses.append(sum(epoch_losses) / len(epoch_losses))

# 误差可视化
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# 评估
with torch.no_grad():
    for batch in test_data:
        text, labels = batch.text, batch.label
        predictions = model(text).squeeze(1)
        # 根据您的需求执行评估
