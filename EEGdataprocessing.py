import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
data_path = "EEG.machinelearing_data_BRMH.csv/EEG.machinelearing_data_BRMH.csv"
dataset = pd.read_csv(data_path)

val_size = 0.2
random_state = 42
# label_to_index = {"Addictive disorder": 0, "Trauma and stress related disorder": 1,
#                   "Mood disorder": 2, "Healthy control": 3,
#                   "Obsessive compulsive disorder": 4, "Schizophrenia": 5,
#                   "Anxiety disorder": 6}  # 标签映射字典
#
# for label_name in dataset:
#     if label_name not in label_to_index:
#         label_to_index[label_name] = len(label_to_index)


#确实信息取均值填充
dataset.fillna(dataset.mean(), inplace=True)

# #性别和标签信息数字化
# label_encoder = LabelEncoder()
# dataset['gender_encoded'] = label_encoder.fit_transform(dataset["sex"])
# dataset["label_encoded"] = label_encoder.fit_transform(dataset["label"])

#分离feature和label
data = dataset.drop("label", axis=1).values
labels = dataset["label"].values

# #stadardscaler
# scaler = StandardScaler()
# data_normalized = scaler.fit_transform(data)



# # 定义标签到索引的映射
# label_to_index = {"Addictive disorder", "Trauma and stress related disorder",
#                   "Mood disorder", "Healthy control",
#                   "Obsessive compulsive disorder", "Schizophrenia",
#                   "Anxiety disorder"}

X = np.array(data)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=random_state
    )
label_map = labels

np.savez_compressed('dataset_eeg.npz',
                   X_train=X_train, X_val=X_val,
                   y_train=y_train, y_val=y_val,
                   label_map=label_map)