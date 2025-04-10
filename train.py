import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import HybridSNN
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, functional
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#hyperparameters
learning_rate = 1e-4
num_epochs = 25
batch_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialization
model = HybridSNN().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,                # 增大学习率（原1e-4）
    betas=(0.9, 0.999),
    weight_decay=1e-3
)


data = np.load('dataset_test.npz', allow_pickle=True)

X_train = data['X_train']  # 训练集特征
y_train = data['y_train']  # 训练集标签
X_val = data['X_val']      # 验证集特征
y_val = data['y_val']      # 验证集标签 
train_min = data['X_train'].min()
train_max = data['X_train'].max()

def normalize(x):
    return 2 * (x - train_min) / (train_max - train_min) - 1  # 映射到[-1,1]

X_train = normalize(data['X_train'])
X_val = normalize(data['X_val'])

# 转换为PyTorch Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

print(y_val)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# train
best_val_acc = 0.0
train_losses = []
val_losses = []
val_accuracies = []

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        functional.reset_net(model)  # 重置网络状态
        train_loss += loss.item() * inputs.size(0)
        
    # val
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Trian_loss: {train_loss:.4f} | Val_loss: {val_loss:.4f} | Val_Accuracy {val_acc:.4f}")


model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 收集预测和真实标签
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 转换为 NumPy 数组
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))

# 定义类别标签（根据你的数据集修改）
class_names = ["Class 0", "Class 1", "Class 2", "Class3", "Class4"]  # 替换为你的类别名称

# 绘制热力图
sns.heatmap(
    cm,
    annot=True,
    fmt="d",          # 显示整数
    cmap="Blues",      # 颜色映射
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=False
)

# 添加标题和标签
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)  # 旋转 x 轴标签
plt.yticks(rotation=0)   # 保持 y 轴标签水平

# 保存图片
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png') 
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('val_accuracy.png')  
plt.show()






