import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets.mnist import read_image_file, read_label_file  

# 确保中文显示正常
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  
plt.rcParams["axes.unicode_minus"] = False 

current_dir = os.path.dirname(os.path.abspath(__file__))  
mnist_root = os.path.join(current_dir, '..', 'data')       
raw_dir = os.path.join(mnist_root, 'MNIST', 'raw')       

train_img_file = 'train-images.idx3-ubyte'
train_lbl_file = 'train-labels.idx1-ubyte'
test_img_file = 't10k-images.idx3-ubyte'
test_lbl_file = 't10k-labels.idx1-ubyte'

class CustomMNIST(datasets.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if self.train:
            self.images = read_image_file(os.path.join(raw_dir, train_img_file))
            self.labels = read_label_file(os.path.join(raw_dir, train_lbl_file))
        else:
            self.images = read_image_file(os.path.join(raw_dir, test_img_file))
            self.labels = read_label_file(os.path.join(raw_dir, test_lbl_file))

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = transforms.ToPILImage()(img)  # 转为PIL图像
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)

# 超参数设置
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.005
DECAY_STEPS = 4000
DECAY_RATE = 0.1

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 加载数据集
train_dataset = CustomMNIST(
    root=mnist_root,
    train=True,
    transform=transform
)
test_dataset = CustomMNIST(
    root=mnist_root,
    train=False,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 定义多层感知器模型
class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# 训练过程可视化函数
def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=24)
    plt.xlabel("迭代次数", fontsize=20)
    plt.ylabel("损失/准确率", fontsize=20)
    plt.plot(iters, costs, 'r-', label=label_cost)
    plt.plot(iters, accs, 'g-', label=label_acc)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def draw_process(title, color, iters, data, label):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=24)
    plt.xlabel("迭代次数", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 主训练函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = MultilayerPerceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=DECAY_STEPS, gamma=DECAY_RATE)

    all_train_iter = 0
    all_train_iters = []
    all_train_costs = []
    all_train_accs = []
    best_test_acc = 0.0

    # 模型保存目录
    work_dir = os.path.join(current_dir, '..', 'work')
    os.makedirs(work_dir, exist_ok=True)
    print(f"模型保存目录: {work_dir}\n")

    # 开始训练
    model.train()
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"=== 第 {epoch+1}/{EPOCHS} 轮，学习率: {current_lr:.6f} ===")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / labels.size(0)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 记录训练数据
            all_train_iter += BATCH_SIZE
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.item())
            all_train_accs.append(acc)

            # 打印日志
            if (batch_idx + 1) % 50 == 0:
                print(f"轮次{epoch+1} - 批次{batch_idx+1}: 损失={loss.item():.4f}, 准确率={acc:.4f}")

        # 测试集评估
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

            test_acc = test_correct / test_total
            print(f"\n测试集准确率: {test_acc:.5f}, 最佳准确率: {best_test_acc:.5f}")

            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if epoch > 10:
                    save_path = os.path.join(work_dir, f'mnist_model_epoch_{epoch}.pth')
                    torch.save(model.state_dict(), save_path)
                    print(f"保存最佳模型: {save_path}\n")
            else:
                print()

        model.train()

    final_path = os.path.join(work_dir, 'mnist_final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"训练完成，最终模型: {final_path}")
    print(f"最佳测试准确率: {best_test_acc:.5f}")

    draw_train_process("训练过程", all_train_iters, all_train_costs, all_train_accs, "训练损失", "训练准确率")
    draw_process("训练损失", "red", all_train_iters, all_train_costs, "训练损失")
    draw_process("训练准确率", "green", all_train_iters, all_train_accs, "训练准确率")

if __name__ == '__main__':
    main()