import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  
plt.rcParams["axes.unicode_minus"] = False 

# 定义与训练时一致的多层感知器模型
class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()  # 展平图像(28x28 -> 784)
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

# 图像加载与预处理函数
def load_image(file):
    im = Image.open(file).convert('L')  # 转为灰度图
    im = im.resize((28, 28), Image.Resampling.LANCZOS)  # 高画质resize
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)  # 调整形状
    im = im / 255.0 * 2.0 - 1.0  # 归一化到[-1, 1]
    return im

base_dir = os.path.dirname(os.path.abspath(__file__))
infer_path = os.path.join(base_dir, '..', 'infer_3.png')  
label_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultilayerPerceptron().to(device)
    model_path = os.path.join(base_dir, '..', 'work', 'mnist_model_epoch_22.pth')  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    
    infer_img = load_image(infer_path)
    infer_tensor = torch.from_numpy(infer_img).to(device)

    with torch.no_grad():  
        output = model(infer_tensor)
        pred_label = int(torch.argmax(output, dim=1).item())  

    print(f"预测结果: {label_list[pred_label]}")

if __name__ == '__main__':
    predict()