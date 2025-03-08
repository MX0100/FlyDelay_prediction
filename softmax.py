import numpy as np
import pandas as pd
import os
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.utils import to_categorical

# 关闭 OneDNN 提示（可选）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 读取数据
file_path = "Flight_Data (1).csv"
df = pd.read_csv(file_path)
df["Is Delayed"] = (df["Delay (min)"] > 10).astype(int)

# 选择特征
features = [
    "Is Weekend", "Is Holiday", "Delay (min)",
    "dep_Blowing Snow", "dep_Clear", "dep_Cloudy", "dep_Drizzle", "dep_Fog",
    "dep_Rain", "dep_Snow", "dep_Snow Showers",
    "arr_Blowing Snow", "arr_Clear", "arr_Cloudy", "arr_Drizzle", "arr_Fog",
    "arr_Rain", "arr_Snow", "arr_Snow Showers"
]
target = "Is Delayed"

# 处理数据
X = df[features].dropna().astype(float).values
y = df[target].dropna().astype(int).values

# **转换为 one-hot 编码**
y = to_categorical(y, num_classes=2)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

# 存储所有批次的预测结果
y_pred_prob_list = []
y_true_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # **检查形状**
    print(f"y_train shape: {y_train.shape}")  # 应该是 (None, 2)
    print(f"y_test shape: {y_test.shape}")    # 应该是 (None, 2)

    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(2, activation='softmax')  # **输出两个类别，使用 Softmax**
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    # **优化 predict()，只调用一次**
    y_pred_prob_list.append(model.predict(X_test, batch_size=8))
    y_true_list.append(y_test)

# **拼接所有批次的预测结果**
y_pred_prob = np.concatenate(y_pred_prob_list)
y_test = np.concatenate(y_true_list)

# **取最大概率对应的类别**
y_pred = np.argmax(y_pred_prob, axis=1)

# **确保 y_test_labels 是正确的维度**
if y_test.ndim == 1:
    y_test_labels = y_test
else:
    y_test_labels = np.argmax(y_test, axis=1)

# 计算评估指标
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred)
recall = recall_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred)
auc = roc_auc_score(y_test[:, 1], y_pred_prob[:, 1])  # 计算 AUC 只针对 "延误" 类别

# 打印最终结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# 保存最终模型
model.save("mlp_flight_delay_model.keras")
