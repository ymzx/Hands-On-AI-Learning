import jieba
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.data.utils import get_tokenizer

# 示例数据
data = [
    {"text": "我喜欢吃苹果", "label": "水果"},
    {"text": "今天去了动物园，看到了很多动物", "label": "动物"},
    {"text": "苹果和香蕉都是水果", "label": "水果"},
    {"text": "家里养了一只猫", "label": "动物"},
]

# 存储示例数据到CSV文件
import pandas as pd
df = pd.DataFrame(data)
df.to_csv("text_data.csv", index=False)