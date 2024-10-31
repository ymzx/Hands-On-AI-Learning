import torch
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from train import SimpleRNN  # 导入训练模型
import jieba


def custom_tokenize(text):
    # 使用 jieba 对中文文本进行分词
    return list(jieba.cut(text))


# 数据准备
def load_data():
    TEXT = Field(sequential=True, tokenize=custom_tokenize, lower=True, batch_first=True)
    LABEL = LabelField()

    # 加载数据集
    data_fields = [("text", TEXT), ("label", LABEL)]
    test_data = TabularDataset(path='data/test.csv', format='csv', fields=data_fields, skip_header=True)

    TEXT.build_vocab(test_data)
    LABEL.build_vocab(test_data)

    return test_data, TEXT, LABEL


# 加载模型
def load_model(vocab_size, num_classes, path):
    model = SimpleRNN(vocab_size, embed_size=100, num_classes=num_classes)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


# 测试模型
def test_model(model, iterator):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch.text, batch.label
            predictions = model(text)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')


# 主程序
def main():
    test_data, TEXT, LABEL = load_data()
    test_iterator = BucketIterator(test_data, batch_size=8, shuffle=False)

    model = load_model(vocab_size=len(TEXT.vocab), num_classes=len(LABEL.vocab), path='models/model.pth')

    # 测试模型
    test_model(model, test_iterator)


if __name__ == '__main__':
    main()
