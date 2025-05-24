import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from modules import MyTransforms
from modules.ResNet import ResNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "./datasets/chest_xray/test"
MODEL_PATH = "./models/512/model_epoch_2.pth"
NUM_WORKERS = 2
BATCH_SIZE = 16
RANDOM_SEED = 114514

# 设置随机种子确保可复现性
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def test():
    # 加载模型
    model = ResNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # 数据预处理
    test_transform = MyTransforms.TestCompose()


    # 加载数据集
    test_dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=test_transform
    )

    print(test_dataset.class_to_idx)

    print(f"Number of tseting samples: {len(test_dataset)}")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 测试模型
    model.eval()

    test_total = 0
    test_correct = 0
    test_label_correct = {0:0, 1:0}
    test_label_total = {0:0, 1:0}


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Compute label accuracy
            for label, pred in zip(labels, predicted):
                label = label.item()
                pred = pred.item()
                if label == pred:
                    test_label_correct[label] += 1
                test_label_total[label] += 1 
            print('step: ', i, ', correct:', 100.0 * (predicted == labels).sum().item() / labels.size(0), '%')


    print(f'Test Accuracy: {100. * test_correct / test_total:.2f}%')
    acc = {label: 100.0 * test_label_correct.get(label, 0) / test_label_total.get(label, 1) for label in test_label_correct}
    print(f'Test Label Accuracy: {acc}')

if __name__ == '__main__':
    test()

