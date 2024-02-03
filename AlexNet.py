import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# 选择设备（一般是GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义获取训练集和验证机的数据加载器
def get_train_valid_loader(data_dir,        # 数据集路径
                           batch_size,      # 每批几张图片
                           augment,         # 是否数据增强
                           random_seed,     # 随机种子，用于后续生成随机数，去打乱图片顺序
                           valid_size=0.1,  # 验证集图片数设置为从训练集中抽取其整体数量的 0.1 
                           shuffle=True):   # 是否洗牌（打乱图片顺序），默认为“是”
    
    # 归一化
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # 给定每个通道的均值
        std=[0.2023, 0.1994, 0.2010],   # 给定每个通道的标准差
    )                                   # 可以加速收敛并提高模型泛化能力

    # 定义验证集图像变换
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),   # 将图像resize到227x227
            transforms.ToTensor(),          # 转换为Tensor
            normalize,
    ])
    if augment:     # 是否数据增强
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ToTensor(),                  # 转换为Tensor
            normalize,                              # 归一化
        ])
    else:
        train_transform = transforms.Compose([   
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    # 下载并加载训练集
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    # 下载并加载验证集
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )   # 测试验证集和训练集的图片数量是相同的

    num_train = len(train_dataset)  # 训练集的图片数
    indices = list(range(num_train))    # 生成一个含有0~(num_train-1)所有整数的索引列表
    # 根据验证集尺寸 valid_size × 训练集图片总数 num_train 计算验证集数量（向下取整）
    split = int(np.floor(valid_size * num_train))

    if shuffle:     # 是否打乱
        # 用给定的随机种子 random_seed 生成随机数，这样每次程序都能得到相同的打乱结果
        np.random.seed(random_seed)
        # 对索引列表进行随机打乱
        np.random.shuffle(indices)

    # 训练集：从split开始到末尾结束，验证集：从0开始到split结束结束
    train_idx, valid_idx = indices[split:], indices[:split] 
    # 创建 2 个采样器。这些采样器会在每次迭代时按照给定的索引采样数据，从而形成训练集和验证集的迭代器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 加载数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

# 定义加载测试集的数据加载器
def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    # 下载并加载测试集
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    # 加载测试数据
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

# 数据集下载路径
data_dir = r'D:\pycode\cnn_from_scratch\data'     # 在当前工作文件夹下的data文件夹

# 调用函数，获取训练集和验证集的 DataLoader
train_loader, valid_loader = get_train_valid_loader(data_dir = data_dir, 
                                                    batch_size = 64, 
                                                    augment = True,     # 仅对训练集增强
                                                    random_seed = 1)

# 调用函数，获取测试集的 DataLoader
test_loader = get_test_loader(data_dir = data_dir,
                              batch_size = 64)

# --------------------------------- #
#           构建 AlexNet            #
# --------------------------------- #
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv_block1 = nn.Sequential(
                        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                        nn.BatchNorm2d(96),
                        nn.ReLU()
                        )
        self.conv_block2 = nn.Sequential(
                        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
        self.conv_block3 = nn.Sequential(
                        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(384),
                        nn.ReLU()
                        )
        self.conv_block4 = nn.Sequential(
                        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(384),
                        nn.ReLU()
                        )
        self.conv_block5 = nn.Sequential(
                        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Sequential(nn.Dropout(0.5),nn.Linear(9216, 4096),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(0.5),nn.Linear(4096, 4096),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.pool(out)
        out = self.conv_block2(out)
        out = self.pool(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# 设置超参数
num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

model = AlexNet(num_classes).to(device)

# 设置损失函数
criterion = nn.CrossEntropyLoss()

# 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

# 训练集总批次数，即 1 个 epoch 要训练几个 step
total_step = len(train_loader)
total_step = len(train_loader)

# --------------------------------- #
#             开始训练               #
# --------------------------------- #
# 外部循环，一个 epoch 一个 epoch 地循环
for epoch in range(num_epochs):
    # 内部循环，一个 batch 一个 batch（一个 step 一个 step）地把数据喂给 AlexNet
    for i, (images, labels) in enumerate(train_loader):  
        # 把图像和标签放到设备（通常是GPU）上
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        # 计算输出和标签的损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()   # 在优化器执行更新之前，清零之前的梯度信息，以避免梯度累积
        loss.backward()         # 计算模型参数相对于损失的梯度（求导）
        optimizer.step()        # 这个 step 的数据已经训练完了。执行一步参数更新（优化）

    # 输出训练信息
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # ---------- 验证 ---------- #
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        # 输出验证结果
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 

# --------------------------------- #
#             开始测试               #
# --------------------------------- #
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   