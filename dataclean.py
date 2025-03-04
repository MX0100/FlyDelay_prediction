import pandas as pd

# 读取 CSV 文件（请根据实际情况调整文件路径和编码）
df = pd.read_csv("FullAc697.csv", encoding="utf-8")

# 假设文件中已有两个字段，分别为出发天气和到达天气
# 如实际字段名不同，请修改下列字段名
dep_weather_col = "YYT weather"
arr_weather_col = "YYZ weather"

# 定义所有可能的天气条件
weather_conditions = ["Blowing Snow", "Clear", "Cloudy", "Drizzle", "Fog", "Rain", "Snow", "Snow Showers"]

# 为出发天气创建对应二值列
for condition in weather_conditions:
    col_name = f"dep_{condition}"
    # 判断出发天气是否等于当前天气条件，若是则为 1，否则为 0
    df[col_name] = (df[dep_weather_col] == condition).astype(int)

# 为到达天气创建对应二值列
for condition in weather_conditions:
    col_name = f"arr_{condition}"
    df[col_name] = (df[arr_weather_col] == condition).astype(int)

# 如果需要，可以将处理结果保存为新的 CSV 文件
df.to_csv("FullAc697_processed.csv", index=False)
