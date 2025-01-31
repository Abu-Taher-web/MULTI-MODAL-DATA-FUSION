# 🚀 Multi-Modal Physical Exercise Classification Project

## 📌 Project Overview
This project focuses on classifying physical exercises using multi-modal data (accelerometer and depth camera) from the **MEx dataset** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MEx#)). The goal is to develop user-independent models to recognize 7 types of exercises performed by subjects lying on a mat. The project explores data fusion techniques at different levels (unimodal, feature-level, decision-level) and includes a bonus task for biometric identification.

### ✨ Key Features
- **📊 Data Preprocessing**: Windowing of raw data into 5-second sequences with 3-second overlap.
- **🔀 Multi-Modal Fusion**: Techniques to combine accelerometer and depth camera data.
- **🤖 Classification**: Evaluation using confusion matrices and F1 scores.
- **📷 Visualization**: Examples of accelerometer time-series and depth camera images.
- **🆔 Biometric Identification**: Optional task to identify subjects from the dataset.

---

## 📂 Dataset
A subset of the **MEx dataset** is used, containing:
- **📡 Accelerometer Data**: 500x3 matrix (3-axis measurements from a thigh sensor).
- **📸 Depth Camera Data**: 5x192 matrix (aerial view frames).

The dataset is preprocessed into shorter sequences to generate training/testing examples.  
📁 **Dataset Location**: Provided as a subset in the project files (not included in this repository).

---

## 📑 Project Structure
The workflow is divided into phases:

### 1️⃣ **Data Preparation, Exploration, and Visualization**
  - 📥 Loading raw data from CSV files.
  - ✂️ Windowing long sequences into 5-second segments.
  - 📊 Visualization of accelerometer time-series and depth images.

### 2️⃣ **Unimodal Fusion for Classification**
  - 🏷️ Feature extraction for accelerometer and depth camera data separately.
  - 🏆 Classification using models like KNN, LDA, or PCA.

### 3️⃣ **Feature-Level Fusion for Multimodal Classification**
  - 🔗 Combining features from both modalities before classification.

### 4️⃣ **Decision-Level Fusion for Multimodal Classification**
  - 🏗️ Aggregating predictions from unimodal models.

### 5️⃣ **Bonus Task: Multimodal Biometric Identification**
  - 🆔 Identifying individuals using sensor data.

---

## 🔬 Methods & Techniques Used
### 🛠️ Data Preprocessing
- 📏 **Feature scaling** using `StandardScaler` and `MinMaxScaler`.
- 🔠 **Encoding categorical variables** using `LabelEncoder`.
- 📉 **Dimensionality reduction** via `PCA` and `LDA`.

### 🧠 Machine Learning Models Implemented
- 🤖 **Support Vector Machines (SVM)** (`SVC`)
- 🔍 **K-Nearest Neighbors (KNN)** (`KNeighborsClassifier`)
- 🧮 **Naïve Bayes Classifier** (`GaussianNB`)
- ⚡ **Adaptive Boosting (AdaBoost)** (`AdaBoostClassifier`)
- 🎯 **Calibrated Classifier** (`CalibratedClassifierCV`)

### 📈 Model Evaluation
- 📊 **Confusion Matrix** (`confusion_matrix` from `sklearn`)
- 🔎 **Grid Search for Hyperparameter Tuning** (`GridSearchCV`)

## 🔗 Dependencies
Ensure you have the following Python libraries installed before running the notebook:

```bash
pip install numpy pandas matplotlib scipy scikit-learn pillow
```

## 🏃 Usage
Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/Abu-Taher-web/MULTI-MODAL-DATA-FUSION.git
cd MULTI-MODAL-DATA-FUSION
```

Run the Jupyter Notebook:

```bash
jupyter notebook Final_Group_project_work_MEx_classification_Submission.ipynb
```

## 🎯 Results
- ✅ The project successfully applies **multi-modal data fusion** techniques to classify physical exercises.
- 📊 Various **machine learning models** are compared, and the best-performing model is identified.
- 🏆 Feature engineering and **dimensionality reduction** improve classification accuracy.

## 👨‍💻 Contributors
- **📝 Abu Taher**
- **📝 Md. Rabiul Hasan**

## 📜 License
This project is released under the **MIT License**.

---

