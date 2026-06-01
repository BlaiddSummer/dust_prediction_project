import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import pickle
import os

# 定义粉尘浓度的BP神经网络模型
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_size3, 6)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 定义B值预测的神经网络模型
class EnhancedBPNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(EnhancedBPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_size3, 1)  # 输出为B值
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 数据清理：去除异常值
def remove_outliers(input_data, output_data, threshold=3):
    combined_data = np.hstack((input_data, output_data))
    z_scores = np.abs((combined_data - np.mean(combined_data, axis=0)) / np.std(combined_data, axis=0))
    filtered_data = combined_data[(z_scores < threshold).all(axis=1)]
    return filtered_data[:, :-output_data.shape[1]], filtered_data[:, -output_data.shape[1]:]

# 数据增强：增加随机噪声
def augment_data(input_data, output_data, noise_level=0.01, n_augmentations=5):
    augmented_input_data = []
    augmented_output_data = []
    for _ in range(n_augmentations):
        noise = np.random.normal(0, noise_level, input_data.shape)
        augmented_input = input_data + noise
        augmented_input_data.append(augmented_input)
        augmented_output_data.append(output_data)
    return np.vstack(augmented_input_data), np.vstack(augmented_output_data)

# 训练单个模型的函数
def train_single_model(model, optimizer, criterion, input_train, output_train, input_test, output_test, max_epochs, patience):
    best_loss = float('inf')
    trigger_times = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()
        outputs = model(input_train)
        loss = criterion(outputs, output_train)
        loss.backward()
        optimizer.step()

        # 验证模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(input_test)
            val_loss = criterion(val_outputs, output_test)

        # 检查早停条件
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1

        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}, best validation loss: {best_loss:.6f}")
            break

    # 恢复最佳模型状态
    model.load_state_dict(best_model_state)
    return model, best_loss

def train_models(dust_file='粉尘数据all.csv', b_file='粉尘数据-b.csv', save_dir='./models', max_epochs=5000, patience=200):
    os.makedirs(save_dir, exist_ok=True)

    # 定义列顺序
    dust_input_columns = ['dmmj', 'jzb', 'cf', 'kcjl', 'yf', 'xfk', 'fbft']
    dust_output_columns = ['0m', '5m', '10m', '15m', '20m', '25m']
    b_input_columns = ['dmmj', 'jzb', 'cf', 'kcjl', 'yf', 'xfk', 'fbft']
    b_output_column = 'b'

    # 读取数据
    data = pd.read_csv(dust_file)
    dust_input_data = data[dust_input_columns].values
    dust_output_data = data[dust_output_columns].values

    data_b = pd.read_csv(b_file)
    b_input_data = data_b[b_input_columns].values
    b_output_data = data_b[[b_output_column]].values

    # 数据清理和增强
    dust_input_data, dust_output_data = remove_outliers(dust_input_data, dust_output_data)
    b_input_data, b_output_data = remove_outliers(b_input_data, b_output_data)

    dust_input_data = np.hstack([dust_input_data, dust_input_data ** 2])
    b_input_data = np.hstack([b_input_data, b_input_data ** 2])

    dust_input_aug, dust_output_aug = augment_data(dust_input_data, dust_output_data)
    b_input_aug, b_output_aug = augment_data(b_input_data, b_output_data)

    dust_input_data = np.vstack([dust_input_data, dust_input_aug])
    dust_output_data = np.vstack([dust_output_data, dust_output_aug])
    b_input_data = np.vstack([b_input_data, b_input_aug])
    b_output_data = np.vstack([b_output_data, b_output_aug])

    # 数据归一化
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    dust_input_normalized = scaler_x.fit_transform(dust_input_data)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    dust_output_normalized = scaler_y.fit_transform(dust_output_data)

    scaler_x_b = MinMaxScaler(feature_range=(0, 1))
    b_input_normalized = scaler_x_b.fit_transform(b_input_data)

    # 保存归一化器
    with open(os.path.join(save_dir, 'scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(save_dir, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(save_dir, 'scaler_x_b.pkl'), 'wb') as f:
        pickle.dump(scaler_x_b, f)



    # 初始化模型参数
    dust_input_size = dust_input_normalized.shape[1]
    b_input_size = b_input_normalized.shape[1]

    # 损失函数
    criterion = nn.SmoothL1Loss()
    kf_dust = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_b = KFold(n_splits=5, shuffle=True, random_state=42)

    # 训练粉尘浓度模型
    for fold, (train_index, test_index) in enumerate(kf_dust.split(dust_input_normalized)):
        print(f"Fold {fold + 1} - 粉尘浓度模型训练")
        model = BPNeuralNetwork(dust_input_size, 64, 32, 16)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        input_train = torch.Tensor(dust_input_normalized[train_index])
        output_train = torch.Tensor(dust_output_normalized[train_index])
        input_test = torch.Tensor(dust_input_normalized[test_index])
        output_test = torch.Tensor(dust_output_normalized[test_index])

        model, best_loss = train_single_model(model, optimizer, criterion, input_train, output_train, input_test, output_test, max_epochs, patience)
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_dust_fold_{fold + 1}.pth'))
        print(f"Fold {fold + 1} completed, best validation loss: {best_loss:.6f}")

    # 训练 B 值模型
    for fold, (train_index, test_index) in enumerate(kf_b.split(b_input_normalized)):
        print(f"Fold {fold + 1} - B 值模型训练")
        model_b = EnhancedBPNeuralNetwork(b_input_size, 128, 64, 32)
        optimizer_b = optim.Adam(model_b.parameters(), lr=0.001, weight_decay=1e-4)

        input_train_b = torch.Tensor(b_input_normalized[train_index])
        output_train_b = torch.Tensor(b_output_data[train_index])
        input_test_b = torch.Tensor(b_input_normalized[test_index])
        output_test_b = torch.Tensor(b_output_data[test_index])

        model_b, best_loss_b = train_single_model(model_b, optimizer_b, criterion, input_train_b, output_train_b, input_test_b, output_test_b, max_epochs, patience)
        torch.save(model_b.state_dict(), os.path.join(save_dir, f'model_b_fold_{fold + 1}.pth'))
        print(f"Fold {fold + 1} completed, best validation loss: {best_loss_b:.6f}")
        

