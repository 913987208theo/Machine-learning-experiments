import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F


# 定义 ResNet-50 模型，包含多实例聚合层
class MILResNet50(nn.Module):
    def __init__(self):
        super(MILResNet50, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 最后一层输出二分类

    def forward(self, x):
        # 输入形状为 [batch_size * bag_size, channels, height, width]
        instance_preds = self.resnet(x)  # 每个实例通过 ResNet50 计算
        instance_preds = instance_preds.view(-1, bag_size)  # 将输出 reshape 成 [batch_size, bag_size]

        # 输出每个实例的预测概率，而不是包的聚合
        return torch.sigmoid(instance_preds)  # 返回每个实例的预测概率


# 数据预处理和归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet需要较大的输入尺寸
    transforms.ToTensor(),
    #对于一个 RGB 图像，原始像素值范围通常为 [0, 255]。ToTensor() 会将这些值除以 255，使它们缩放到 [0, 1] 的范围。
    #转换后的张量维度通常是 [C, H, W]，即通道数（Channels）、高度（Height）、宽度（Width）。例如，对于 RGB 图像，通道数 C 为 3（红、绿、蓝通道）。
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #对图像的每个通道进行归一化处理，使图像的像素值范围从 [0, 1] 转换为 [-1, 1]。这是通过对图像每个通道进行减去均值并除以标准差的方式来实现的。
])


# 函数：创建包（bags），每个包中包含多个实例
def create_bag(dataset, bag_size=5):
    bags = []
    labels = []

    #将索引随机打乱。这样确保每次生成的包都是不同的，数据集是随机化的。
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in range(0, len(dataset), bag_size):
        # 获取包内的实例
        bag_indices = indices[i:i + bag_size] #表示从 indices 中截取大小为 bag_size 的连续索引，作为当前包的实例索引。

        # 如果包内实例数量小于 bag_size，则忽略该包
        if len(bag_indices) < bag_size:
            continue

        bag = [dataset[idx][0] for idx in bag_indices]
        #使用从 bag_indices 中获取的索引，逐个从 dataset 中提取图像。dataset[idx] 返回一个二元组 (图像, 标签)，[0] 用于只提取图像部分。
        bag_label = max(dataset[idx][1] for idx in bag_indices)  # 如果有狗（正例），标签为 1

        bags.append(torch.stack(bag))  # 将 bag 转换为张量
        labels.append(bag_label)  # 包的标签（是否含有狗）

    return bags, labels


# 函数：添加噪声标签
def add_label_noise(dataset, noise_ratio):
    dog_idx = dataset.class_to_idx['dogs']
    cat_idx = dataset.class_to_idx['cats']

    dog_samples = [i for i, (_, label) in enumerate(dataset.imgs) if label == dog_idx]
    noisy_dog_samples = random.sample(dog_samples, int(noise_ratio * len(dog_samples)))

    for i in noisy_dog_samples:
        dataset.imgs[i] = (dataset.imgs[i][0], cat_idx)  # 将狗标记为猫（添加噪声）

    return dataset


# 函数：评估模型，使用多实例学习
def evaluate_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for bags, labels in test_loader:
            bags = torch.cat(list(bags), dim=0)  # 将包内的实例拉成 4D 张量
            labels = labels.clone().detach().float().to(device)  # 包标签
            bags = bags.to(device)
            outputs = model(bags)

            # 包标签：预测包中是否有狗
            predicted = (outputs.mean(dim=1) > 0.5).float()
            correct += (predicted.view(-1) == labels).sum().item()
            total += labels.size(0)  # 计算总包数

    accuracy = 100 * correct / total
    return accuracy


# 主程序
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
bag_size = 3  # 定义包的大小
test_data = ImageFolder(root='../CatsvsDogs/test', transform=transform)

# 创建包并加载到数据加载器中
test_bags, test_labels = create_bag(test_data, bag_size=bag_size)
test_loader = DataLoader(list(zip(test_bags, test_labels)), batch_size=16, shuffle=False)

# 创建文件用于保存测试结果，并写入表头
with open("accuracy_results_mil_resnet.txt", "w") as f:
    f.write("Noise Ratio (%) \t Test Accuracy (%)\n")

# 噪声率从0%到50%，每次增加1%
for noise_ratio in range(0, 51, 1):
    noise_ratio /= 100  # 将百分比转换为小数

    print(f"\n噪声率：{noise_ratio * 100}%")

    # 加载训练数据集并添加噪声标签
    train_data = ImageFolder(root='../CatsvsDogs/train', transform=transform)
    train_data = add_label_noise(train_data, noise_ratio)
    train_bags, train_labels = create_bag(train_data, bag_size=bag_size)
    train_loader = DataLoader(list(zip(train_bags, train_labels)), batch_size=16, shuffle=True)

    # 创建 ResNet-50 多实例学习模型
    model = MILResNet50().to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    # 定义不同的学习率

    # 提取除 fc 层以外的参数
    feature_params = [param for name, param in model.resnet.named_parameters() if "fc" not in name]

    # 获取 fc 层的参数
    classifier_params = model.resnet.fc.parameters()

    optimizer = optim.Adam([
        {'params': feature_params, 'lr': 1e-4},  # 特征提取部分的学习率
        {'params': classifier_params, 'lr': 1e-3}  # 分类器部分的学习率
    ])

    # 训练模型
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 将包内的实例展开为 4D 张量，并将包标签扩展为实例标签
        for bags, labels in train_loader:
            bags = torch.cat(list(bags), dim=0) #将当前 batch 中的包展开为一个长张量，形状为 [batch_size * bag_size, channels, height, width]，
            labels = torch.tensor(labels).float().to(device)
            bags = bags.to(device)

            # 注意这里，将 labels 的维度调整为 [batch_size]，而不是 [batch_size, 1]
            labels = labels.float().view(-1)  # 改为 (-1)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(bags)
            loss = criterion(outputs.mean(dim=1), labels)  # 使用包级别的预测进行损失计算

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs.mean(dim=1) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] 完成，平均损失：{running_loss / total:.4f}，准确率：{accuracy:.2f}%")

    # 测试模型
    test_accuracy = evaluate_model(test_loader, model)
    print(f"噪声率 {noise_ratio * 100}% 测试集准确率：{test_accuracy:.2f}%")

    # 将测试结果写入文件
    with open("accuracy_results_mil_resnet.txt", "a") as f:
        f.write(f"{noise_ratio * 100:.2f}\t\t{test_accuracy:.2f}\n")
