import torch
import os
import numpy as np
import pandas as pd
import pickle
from pyswarm import pso

# 加载scaler
def load_scalers(save_dir='./models'):
    scaler_x_path = os.path.join(save_dir, 'scaler_x.pkl')
    scaler_y_path = os.path.join(save_dir, 'scaler_y.pkl')
    scaler_x_b_path = os.path.join(save_dir, 'scaler_x_b.pkl')

    with open(scaler_x_path, 'rb') as f:
        scaler_x = pickle.load(f)

    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    with open(scaler_x_b_path, 'rb') as f:
        scaler_x_b = pickle.load(f)

    return scaler_x, scaler_y, scaler_x_b

# 定义粉尘浓度的BP神经网络模型
class BPNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.dropout = torch.nn.Dropout(0.3)  # 添加Dropout层
        self.fc4 = torch.nn.Linear(hidden_size3, 6)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 定义B值预测的神经网络模型
class EnhancedBPNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(EnhancedBPNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(hidden_size3, 1)  # 输出为B值
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 加载模型
def load_models(fold=1, model_dir='./models'):
    model_dust_path = os.path.join(model_dir, f'model_dust_fold_{fold}.pth')
    model_b_path = os.path.join(model_dir, f'model_b_fold_{fold}.pth')

    # 定义模型结构
    model_dust = BPNeuralNetwork(input_size=14, hidden_size1=64, hidden_size2=32, hidden_size3=16)
    model_b = EnhancedBPNeuralNetwork(input_size=14, hidden_size1=128, hidden_size2=64, hidden_size3=32)

    # 检查模型文件是否存在
    if not os.path.exists(model_dust_path):
        raise FileNotFoundError(f"粉尘浓度模型文件未找到: {model_dust_path}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"B值模型文件未找到: {model_b_path}")

    # 加载模型权重
    model_dust.load_state_dict(torch.load(model_dust_path))
    model_dust.eval()

    model_b.load_state_dict(torch.load(model_b_path))
    model_b.eval()

    return model_dust, model_b


# 预测粉尘浓度
def predict_dust_concentration(model_dust, scaler_x, scaler_y, dmmj, yf, cf, xfk, fbft, kcjl, jzb):
    # 构造输入特征并增加平方项
    input_sample = np.array([dmmj, yf, cf, xfk, fbft, kcjl, jzb])
    input_sample_extended = np.hstack([input_sample, input_sample ** 2])
    input_normalized = scaler_x.transform(input_sample_extended.reshape(1, -1))
    input_tensor = torch.Tensor(input_normalized)
    with torch.no_grad():
        prediction_normalized = model_dust(input_tensor)
    prediction = scaler_y.inverse_transform(prediction_normalized.numpy().reshape(1, -1))
    return prediction


# 预测B值
def predict_b_value(model_b, scaler_x_b, dmmj, yf, cf, xfk, fbft, kcjl, jzb):
    # 构造输入特征并增加平方项
    input_values_b = np.array([dmmj, yf, cf, xfk, fbft, kcjl, jzb])
    input_values_b_extended = np.hstack([input_values_b, input_values_b ** 2])
    input_normalized_b = scaler_x_b.transform(input_values_b_extended.reshape(1, -1))
    input_tensor_b = torch.Tensor(input_normalized_b)
    with torch.no_grad():
        prediction_b = model_b(input_tensor_b).item()
    # 使用绝对值确保 B 值为正
    return abs(prediction_b)

# 计算A值
def calculate_A_value(predicted_concentration):
    di_mean = np.mean(predicted_concentration, axis=0)
    di_mean = np.where(di_mean < 0, 0, di_mean)
    di_total = np.sum(di_mean)
    if di_total == 0:
        return 0
    A_value = np.sum(di_mean ** 2) / di_total
    return A_value

# 计算C值
def calculate_C_value(a_value, b_value):
    c_value = 100 * b_value * 0.4 + a_value * 0.6
    return c_value

def get_user_input():
    print("请按照提示输入粉尘浓度模型的以下参数：")
    prompts = [
        "巷道面积 (m²): ",
        "压入风量 (m³/min): ",
        "抽出风量 (m³/min): ",
        "抽尘口距迎头距离 (m): ",
        "控尘装置长度 (m): ",
        "控尘装置到迎头距离 (m): ",
        "径轴向风量比: "
    ]
    keys = ['dmmj', 'yf', 'cf', 'xfk', 'fbft', 'kcjl', 'jzb']
    inputs = []
    for prompt in prompts:
        while True:
            try:
                value = float(input(prompt))
                inputs.append(value)
                break
            except ValueError:
                print("无效输入，请输入有效的数值！")
    return dict(zip(keys, inputs))



# 目标函数，用于PSO优化
def objective_function(model_dust, scaler_x, scaler_y, dmmj, jzb, cf, kcjl, yf, xfk, fbft):
    concentration = predict_dust_concentration(model_dust, scaler_x, scaler_y, dmmj, jzb, cf, kcjl, yf, xfk, fbft)
    A_value = calculate_A_value(concentration)
    return -A_value  # PSO是最小化问题，因此返回负值

# 优化过程
def optimize_and_predict():

    # 加载模型和 scaler
    scaler_x, scaler_y, scaler_x_b = load_scalers()
    model_dust, model_b = load_models()

    # 从用户输入获取初始值
    print("请输入初始参数：")
    last_row = get_user_input()  # 调用通用的输入函数
    dmmj = last_row['dmmj']
    initial_yf = last_row['yf']
    initial_cf = last_row['cf']
    initial_xfk = last_row['xfk']
    initial_fbft = last_row['fbft']
    initial_kcjl = last_row['kcjl']
    initial_jzb = last_row['jzb']

    # 参数范围定义
    jzb_range = (0.5, 1.5)
    cf_range = (350, 600)
    kcjl_range = (10, 20)

    concentration_values = []
    A_values = []
    steps = ['初始值']

    # 初始粉尘浓度预测
    initial_concentration = predict_dust_concentration(
        model_dust, scaler_x, scaler_y,
        dmmj, initial_jzb, initial_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    concentration_values.append(initial_concentration)
    A_values.append(calculate_A_value(initial_concentration))

    # 初始 B 值和 C 值计算
    initial_b_value = predict_b_value(
        model_b, scaler_x_b,
        dmmj, initial_jzb, initial_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    initial_C_value = calculate_C_value(A_values[-1], initial_b_value)

    print(f"巷道面积: {dmmj} 平方米 - 初始粉尘浓度预测:")
    for i, dist in enumerate([0, 5, 10, 15, 20, 25]):
        print(f"距离 {dist}m: {initial_concentration[0][i]:.4f} mg/m3")
    print(f"初始参数: 径轴向风量比={initial_jzb:.2f}, 抽出风量={initial_cf:.2f}, "
          f"控尘装置到迎头距离={initial_kcjl:.2f}, 初始A值={A_values[-1]:.2f}, 初始C值={initial_C_value:.2f}")

    # 优化径轴向风量比
    def optimize_jzb(x):
        return -objective_function(
            model_dust, scaler_x, scaler_y,
            dmmj, x[0], initial_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
        )

    best_jzb, _ = pso(optimize_jzb, [jzb_range[0]], [jzb_range[1]], swarmsize=50, maxiter=100)
    best_jzb = float(best_jzb)
    concentration_after_jzb = predict_dust_concentration(
        model_dust, scaler_x, scaler_y,
        dmmj, best_jzb, initial_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    A_value_after_jzb = calculate_A_value(concentration_after_jzb)
    current_B_value = predict_b_value(
        model_b, scaler_x_b,
        dmmj, best_jzb, initial_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    current_C_value = calculate_C_value(A_value_after_jzb, current_B_value)
    A_values.append(A_value_after_jzb)
    steps.append('优化径轴向风量比')

    print(f"\n优化径轴向风量比后的粉尘浓度:")
    for i, dist in enumerate([0, 5, 10, 15, 20, 25]):
        print(f"距离 {dist}m: {concentration_after_jzb[0][i]:.4f} mg/m3")
    print(f"优化后径轴向风量比: {best_jzb:.2f}, 当前A值: {A_value_after_jzb:.2f}, 当前C值: {current_C_value:.2f}")

    # 优化抽出风量
    def optimize_cf(x):
        return -objective_function(
            model_dust, scaler_x, scaler_y,
            dmmj, best_jzb, x[0], initial_kcjl, initial_yf, initial_xfk, initial_fbft
        )

    best_cf, _ = pso(optimize_cf, [cf_range[0]], [cf_range[1]], swarmsize=50, maxiter=100)
    best_cf = float(best_cf)
    concentration_after_cf = predict_dust_concentration(
        model_dust, scaler_x, scaler_y,
        dmmj, best_jzb, best_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    A_value_after_cf = calculate_A_value(concentration_after_cf)
    current_B_value = predict_b_value(
        model_b, scaler_x_b,
        dmmj, best_jzb, best_cf, initial_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    current_C_value = calculate_C_value(A_value_after_cf, current_B_value)
    A_values.append(A_value_after_cf)
    steps.append('优化抽出风量')

    print(f"\n优化抽出风量后的粉尘浓度:")
    for i, dist in enumerate([0, 5, 10, 15, 20, 25]):
        print(f"距离 {dist}m: {concentration_after_cf[0][i]:.4f} mg/m3")
    print(f"优化后抽出风量: {best_cf:.2f}, 当前A值: {A_value_after_cf:.2f}, 当前C值: {current_C_value:.2f}")

    # 优化控尘装置到迎头距离
    def optimize_kcjl(x):
        return -objective_function(
            model_dust, scaler_x, scaler_y,
            dmmj, best_jzb, best_cf, x[0], initial_yf, initial_xfk, initial_fbft
        )

    best_kcjl, _ = pso(optimize_kcjl, [kcjl_range[0]], [kcjl_range[1]], swarmsize=50, maxiter=100)
    best_kcjl = float(best_kcjl)
    concentration_after_kcjl = predict_dust_concentration(
        model_dust, scaler_x, scaler_y,
        dmmj, best_jzb, best_cf, best_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    A_value_after_kcjl = calculate_A_value(concentration_after_kcjl)
    current_B_value = predict_b_value(
        model_b, scaler_x_b,
        dmmj, best_jzb, best_cf, best_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    current_C_value = calculate_C_value(A_value_after_kcjl, current_B_value)
    A_values.append(A_value_after_kcjl)
    steps.append('优化控尘装置到迎头距离')

    print(f"\n优化控尘装置到迎头距离后的粉尘浓度:")
    for i, dist in enumerate([0, 5, 10, 15, 20, 25]):
        print(f"距离 {dist}m: {concentration_after_kcjl[0][i]:.4f} mg/m3")
    print(f"优化后控尘装置到迎头距离: {best_kcjl:.2f}, 当前A值: {A_value_after_kcjl:.2f}, 当前C值: {current_C_value:.2f}")

    # 最终优化结果
    final_A_value = A_values[-1]
    final_B_value = predict_b_value(
        model_b, scaler_x_b,
        dmmj, best_jzb, best_cf, best_kcjl, initial_yf, initial_xfk, initial_fbft
    )
    final_C_value = calculate_C_value(final_A_value, final_B_value)

    print(f"\n巷道面积 {dmmj} 平方米的最终优化结果:")
    print(f"最优径轴向风量比: {best_jzb:.2f}")
    print(f"最优抽出风量: {best_cf:.2f}")
    print(f"最优控尘装置到迎头距离: {best_kcjl:.2f}")
    print(f"最终A值: {final_A_value:.2f}")
    print(f"最终B值: {final_B_value:.2f}")
    print(f"最终C值: {final_C_value:.2f}")


