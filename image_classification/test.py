import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import SimpleCNN  # 导入模型定义


# 数据处理和加载
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.ImageFolder(root='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
num_classes = len(test_dataset.classes)
model = SimpleCNN(num_classes)
model_path = "./models/model_batch_20.pth"  # 指定要加载的模型路径


# 测试函数
def test(model, model_path, test_loader):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    # 测试模型
    test(model, model_path, test_loader)
