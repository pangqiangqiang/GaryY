import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd

# 读取 test.csv 文件
dataset = pd.read_csv("C:/Users/29508/Desktop/test.csv", encoding='Windows-1252')

# 检查数据框架中的实际列名
print(dataset.columns)

# 定义清理和标记化函数
def clean_and_tokenize(text):
    # 检查 text 是否是缺失值
    if isinstance(text, str):  # 如果 text 是字符串
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        return tokens
    else:
        # 如果 text 是缺失值，你可以在这里进行处理，比如返回一个默认值或者忽略
        return []

# 对 'text' 列进行清理和标记化
dataset['text'] = dataset['text'].apply(clean_and_tokenize)

# 移除停用词
stop_words = set(stopwords.words('english'))
dataset['text'] = dataset['text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# 词干化
stemmer = PorterStemmer()
dataset['text'] = dataset['text'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# 打印处理后的数据框架
print(dataset)


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

class Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_dataset()

    def load_dataset(self):
        dataset = pd.read_csv(self.file_path, encoding='Windows-1252')
        self.dataset = self.preprocess_dataset(dataset)

    def preprocess_dataset(self, dataset):
        # 在这里实现你的预处理逻辑
        # ...

        return dataset
