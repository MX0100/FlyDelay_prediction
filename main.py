import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the cleaned data
file_path = "Flight_Data (1).csv"
df = pd.read_csv(file_path)
df["Is Delayed"] = (df["Delay (min)"] > 10).astype(int)

# Select features for MLP
weather_features = [
    "dep_Blowing Snow", "dep_Clear", "dep_Cloudy", "dep_Drizzle", "dep_Fog",
    "dep_Rain", "dep_Snow", "dep_Snow Showers",
    "arr_Blowing Snow", "arr_Clear", "arr_Cloudy", "arr_Drizzle", "arr_Fog",
    "arr_Rain", "arr_Snow", "arr_Snow Showers"
]

features = ["Is Weekend", "Is Holiday", "Delay (min)"] + weather_features
target = "Is Delayed"

# Filter valid data
X = df[features].dropna().astype(float).values
y = df[target].dropna().astype(int).values

# Standardize data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Record metrics
accuracies, precisions, recalls, f1_scores = [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # MLP model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),       # input_shape=(X.shape[1],) 可以删了
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')     # why save sigmoid 输出层应该有两个神经元 我这只有一个 用一个softmap
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # 为什么用这三种算法不用别的
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)     # 这些数据值是怎么确定的？

    # Predict
    y_pred = (model.predict(X_test) > 0.5).astype(int)      # 0.5是什么意思？你这行写错了 和46行是一起的 分类问题不是这么写的

    # Calculate metrics
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Print average metrics
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f}")

model.save("mlp_flight_delay_model.h5")