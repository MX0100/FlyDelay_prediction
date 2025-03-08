# ✈️ Flight Delay Prediction using MLP

## 📌 Overview
This project predicts whether a flight will be delayed based on various factors such as weather conditions, holidays, and weekends.  
The model is built using a **Multi-Layer Perceptron (MLP) neural network** with **Softmax activation** for binary classification.

## 📊 Dataset
This dataset contains **real flight data** from **Air Canada** flights operating:
- **YYT (St. John's) → YYZ (Toronto Pearson)**
- **YYT (St. John's) → YHZ (Halifax Stanfield)**

### **Features:**
- **Flight conditions**
  - Weather at **departure** & **arrival** airports
- **Temporal features**
  - Whether the flight is on a **weekend**
  - Whether the flight is on a **holiday**
- **Delay information**
  - Flight delay in minutes (**Delay (min)**)
- **Target Variable:**
  - `Is Delayed` → **Binary classification** (0: Not Delayed, 1: Delayed)

## 🏗 Model Architecture
The model consists of:
- **Input Layer**: Takes the flight features (weather, weekend, holiday, delay)
- **Hidden Layers**:
  - **Dense (32 neurons, ReLU)**
  - **Dense (16 neurons, ReLU)**
- **Output Layer**:
  - **Dense (2 neurons, Softmax activation)**
  - **Predicts the probability of delay vs. no delay**

## 📈 Model Performance
| Metric       | Value  |
|-------------|--------|
| **Accuracy**    | 0.7978 |
| **Precision**   | 0.7250 |
| **Recall**      | 0.5370 |
| **F1 Score**    | 0.6170 |
| **AUC Score**   | 0.7310 |

**📝 Interpretation:**
- The model achieves **~80% accuracy**, meaning it correctly predicts delays in most cases.
- **Precision (0.7250)** suggests that **72.5% of predicted delays are correct**.
- **Recall (0.5370)** indicates that **only 53.7% of actual delays were identified**, meaning there are still some missed delays.
- **AUC Score (0.7310)** shows that the model has a good ability to distinguish between delayed and non-delayed flights.

## 🚀 Installation & Usage

### **1️⃣ Install Dependencies**
Ensure you have **Python 3.8+**, then install the required packages:
```sh
pip install -r requirements.txt
````
```sh
conda activate tf-env
```
```shell
conda deactivate
```