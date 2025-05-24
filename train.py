import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision

from modules import ResNet
from modules import MyTransforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 15

EPOCH_OFFSET = 0

LR = 0.001

DATA_PATH = "./datasets/chest_xray/train"
MODEL_PATH = ""
SAVE_MODEL_PATH = "./models/512"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 114514
NUM_WORKERS = 4
BATCH_SIZE = 32

# 设置随机种子确保可复现性
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == '__main__':
    print('device:', DEVICE)

    # 数据预处理
    dataset_transform = MyTransforms.TrainCompose()


    # 加载数据集
    dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=dataset_transform
    )

    # 标签字典
    label_dict = dataset.class_to_idx
    # 反转字典
    label_dict = {v: k for k, v in label_dict.items()}

    print(label_dict)

    # 各类别权重，用以解决样本不平衡问题
    class_weights = torch.tensor([len(dataset) / dataset.targets.count(0), len(dataset) / dataset.targets.count(1)]).to(DEVICE)

    print('class weights:', class_weights)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    print(f"Number of training samples: {len(train_dataset)}")

    
    print(f"Number of validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = ResNet.ResNet().to(DEVICE)

    # 如果指定了模型路径，则加载模型
    if MODEL_PATH is not None and os.path.exists(MODEL_PATH):
        # 加载模型
        print('loading model...')
        print(MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH))

    # 如果模型路径不存在，则初始化模型
    else:
        print('initializing model...')
    
    # 损失函数
    loss_func = nn.CrossEntropyLoss(class_weights)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 变长学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 存储训练和验证的准确率，用于最后打印结果
    train_total_acc = []
    train_label_acc = []
    val_total_acc = []
    val_label_acc = []

    # 创建保存模型的文件夹
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    model.train()
    for epoch in range(EPOCH):
        if optimizer.param_groups[0]['lr'] < 1e-6:
            scheduler.step = lambda: None
            optimizer.param_groups[0]['lr'] = 1e-6
        print('Epoch [{}/{}], lr {}'.format(epoch + 1, EPOCH, optimizer.param_groups[0]['lr']))

        train_correct = 0
        train_total = 0
        label_correct = {0:0, 1:0}
        label_total = {0:0, 1:0}

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            scheduler.step()


            # 统计训练集准确率
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 统计训练集标签准确率
            for label, pred in zip(labels, predicted):
                label = label.item()
                pred = pred.item()
                if label == pred:
                    label_correct[label] += 1
                label_total[label] += 1

            print('step: ', i, ',loss: ', loss.item(), ', correct:', 100.0 * (predicted == labels).sum().item() / labels.size(0), '%')

        train_total_acc.append(100.0 * train_correct / train_total)
        train_label_acc.append({label: 100.0 * label_correct.get(label, 0) / label_total.get(label, 1) for label in label_correct})
        
        # 保存模型
        print(f'Saving model(epoch={epoch})...')

        torch.save(model.state_dict(), SAVE_MODEL_PATH + '/model_epoch_{}.pth'.format(epoch + 1 + EPOCH_OFFSET))

        
        # 验证集准确率
        val_correct = 0
        val_total = 0
        val_label_correct = {0:0, 1:0}
        val_label_total = {0:0, 1:0}

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)

                # 统计验证集准确率
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 统计验证集标签准确率
                for label, pred in zip(labels, predicted):
                    label = label.item()
                    pred = pred.item()
                    if label == pred:
                        val_label_correct[label] += 1
                    val_label_total[label] += 1

        val_total_acc.append(100.0 * val_correct / val_total)
        val_label_acc.append({label: 100.0 * val_label_correct.get(label, 0) / val_label_total.get(label, 1) for label in val_label_correct})

        print(f'Train Accuracy: {train_total_acc[-1]:.2f}%')
        print(f'Validation Accuracy: {val_total_acc[-1]:.2f}%')
        print(f'Train Label Accuracy: {train_label_acc[-1]}')
        print(f'Validation Label Accuracy: {val_label_acc[-1]}')

    # 结果绘制
    epochs = range(1, EPOCH + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_total_acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_total_acc, 'r-', label='Val Acc')
    plt.title('Total Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    for label in [0, 1]:
        train_label = [acc[label] for acc in train_label_acc]
        val_label = [acc[label] for acc in val_label_acc]
        if label == 0:
            train_color = 'b-'
            val_color = 'r--'
        else:
            train_color = 'g-'
            val_color = 'y--'
        plt.plot(epochs, train_label, train_color, label=f'Train Label {label_dict[label]}')
        plt.plot(epochs, val_label, val_color, label=f'Val Label {label_dict[label]}')

    plt.title('Label Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_MODEL_PATH + '/accuracy_plots.png')
    plt.show()