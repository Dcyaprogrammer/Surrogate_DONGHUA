import pandas as pd
import numpy as np

data = pd.read_csv('EEG.machinelearing_data_BRMH.csv')

new_data=data.drop(columns=["education","IQ","Unnamed: 122","no.","sex","age","eeg.date","specific.disorder"])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

PSD = [col for col in new_data.columns if col.startswith("AB")]

filtered_data = new_data[new_data['main.disorder'] != "Mood disorder"].copy()
scaler = StandardScaler()
X = filtered_data[PSD].values
X_scaled = scaler.fit_transform(X)

y = filtered_data['main.disorder']
y_encoded = LabelEncoder().fit_transform(y)


unique_classes = np.unique(y_encoded)
sampling_strategy = {label: 500 for label in unique_classes}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)

balanced_data = pd.DataFrame(X_balanced, columns=PSD)
balanced_data['main.disorder'] = y_balanced
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_new = balanced_data.drop(columns=['main.disorder']).values
y_new = balanced_data['main.disorder'].values


from sklearn.model_selection import train_test_split
from model import ConvAttnModel

model = ConvAttnModel()

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional,surrogate

from torch.utils.data import DataLoader, TensorDataset

#hyperparameters
learning_rate = 1e-4
num_epochs = 1000
batch_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialization
model = ConvAttnModel().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,                
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

X_train, X_val, y_train, y_val = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42)



X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

def replicate_sample(target_classes, num_copies, X_val, y_val):
  num_copies = 1  

  for target_class in target_classes:
    
    mask = y_val == target_class
    X_target = X_val[mask]
    y_target = y_val[mask]

    
    X_replicated = X_target.repeat(num_copies + 1, 1)  
    y_replicated = y_target.repeat(num_copies + 1)

    
    non_target_mask = ~mask
    X_non_target = X_val[non_target_mask]
    y_non_target = y_val[non_target_mask]

    
    X_val_balanced = torch.cat([X_non_target, X_replicated], dim=0)
    y_val_balanced = torch.cat([y_non_target, y_replicated], dim=0)

 
  val_dataset = TensorDataset(X_val_balanced, y_val_balanced)
  val_loader = DataLoader(val_dataset, batch_size=batch_size)

  return val_dataset, val_loader

val_dataset, val_loader = replicate_sample([1,3],1,X_val,y_val)


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
        functional.reset_net(model)  
        train_loss += loss.item() * inputs.size(0)

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
        torch.save(model.state_dict(), 'bbl_best_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Trian_loss: {train_loss:.4f} | Val_loss: {val_loss:.4f} | Val_Accuracy {val_acc:.4f}")


model.load_state_dict(torch.load('bbl_best_model.pth'))
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))


class_names = ["Class 0", "Class 1", "Class 2", "Class3", "Class4", "Class5", "Class6"]  


sns.heatmap(
    cm,
    annot=True,
    fmt="d",          
    cmap="Blues",      
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=False
)


plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)  
plt.yticks(rotation=0)   


plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
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

plt.figure(figsize=(10, 5))


ax1 = plt.gca()
line1, = ax1.plot(train_losses, 'b-', label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)


ax2 = ax1.twinx()
line2, = ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params(axis='y', labelcolor='r')


lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], 
           loc='best',          # 自动选择最佳位置
           facecolor='whitesmoke',
           framealpha=0.7,
           edgecolor='gray')

plt.title('Training Loss and Validation Accuracy')
plt.tight_layout()  
plt.savefig('loss_and_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()


np.savetxt('bbl_train_losses.txt', np.array(train_losses), fmt='%.6f')
np.savetxt('bbl_val_losses.txt', np.array(val_losses), fmt='%.6f')
np.savetxt('bbl_val_accuracies.txt', np.array(val_accuracies), fmt='%.6f')