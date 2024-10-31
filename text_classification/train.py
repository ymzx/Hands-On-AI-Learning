import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import os
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator


# 定义超参数
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 0.001

# 创建保存模型的文件夹
save_path = "./models/"
os.makedirs(save_path, exist_ok=True)


def custom_tokenize(text):
    # 使用 jieba 对中文文本进行分词
    return list(jieba.cut(text))


# 定义模型
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, 64, batch_first=False)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x


# 数据准备
def load_data():
    TEXT = Field(sequential=True, tokenize=custom_tokenize, lower=True, batch_first=True)
    LABEL = LabelField()

    data_fields = [("text", TEXT), ("label", LABEL)]
    train_data, test_data = TabularDataset.splits(
        path='data', train='train.csv', test='test.csv', format='csv',
        fields=data_fields, skip_header=True
    )

    TEXT.build_vocab(train_data, max_size=10000, min_freq=1)
    LABEL.build_vocab(train_data)

    return train_data, test_data, TEXT, LABEL


# 训练模型
def train_model(model, iterator, criterion, optimizer):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch.text, batch.label

        # 确保文本输入和标签都是长整型
        text = text.type(torch.long)
        labels = labels.type(torch.long)

        # 进行模型预测
        predictions = model(text)

        # 打印每个批次的大小和形状
        print(f"Batch size: {text.size(0)}")
        print(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")

        # 计算损失
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 主程序
def main():
    train_data, test_data, TEXT, LABEL = load_data()
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data),
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True)

    model = SimpleRNN(vocab_size=len(TEXT.vocab), embed_size=100, num_classes=len(LABEL.vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练
    for epoch in range(NUM_EPOCHS):
        train_model(model, train_iterator, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} completed.")

    # 保存模型
    save_model(model, 'models/model.pth')
    print("Model saved.")


if __name__ == '__main__':
    main()
