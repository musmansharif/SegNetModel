# Breast Cancer Recurrence Prediction Using Machine Learning

This repository contains the full implementation of a research project focused on predicting breast cancer recurrence using machine learning techniques. Two publicly available datasets, **WDBC (Wisconsin Diagnostic Breast Cancer)** and **WPBC (Wisconsin Prognostic Breast Cancer)**, were used to evaluate the model performance under different experimental settings.

## 📌 Research Objective

To develop, evaluate, and compare a variety of classical and ensemble machine learning classifiers on WDBC and WPBC datasets for the task of breast cancer recurrence prediction — both **with** and **without feature selection** — and to improve upon existing literature using a carefully optimized and reproducible pipeline.

---

## 🧠 Datasets Used

- **WDBC**: Wisconsin Diagnostic Breast Cancer
- **WPBC**: Wisconsin Prognostic Breast Cancer  
All datasets were preprocessed to address missing values, class imbalance, and irrelevant features.

---

## 🧪 Experimental Setup

The study involved two core experimental settings:

### 🔹 Model Setting I — Without Feature Selection
- All features retained (except ID/time)
- Missing values imputed using mean strategy
- Class imbalance handled using **SMOTE**

### 🔹 Model Setting II — With Feature Selection & Scaling
- Feature selection using **SelectKBest** with `mutual_info_classif`
- Top 25 features selected
- Data scaled using **StandardScaler**
- SMOTE applied to balance data

---

## 🤖 Classifiers Implemented

- **Ensemble Models**:  
  `GradientBoosting`, `AdaBoost`, `LightGBM`, `CatBoost`, `XGBoost`, `HistGradientBoosting`, `Stacking`

- **Baseline Models**:  
  `RandomForest`, `SVM`, `LogisticRegression`, `KNN`, `NaiveBayes`, `DecisionTree`

All models were **fine-tuned** for optimal performance.

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Sensitivity (Recall)**
- **Specificity**
- **Precision (PPV)**
- **Negative Predictive Value (NPV)**
- **F1-Score**

---

## 📈 Results Overview

- **WPBC Dataset**:  
  GradientBoosting (with FS): `Accuracy = 93.3%`, `F1-score = 0.939`  
  Outperformed recent studies by **Zuo et al. (2023)** and **Azeroual et al. (2024)**

- **WDBC Dataset**:  
  AdaBoost/LightGBM (with FS): `Accuracy = 98.6%`, `F1-score = 0.987`  
  Improved over past literature using Transformer, LightGBM, and hybrid models

---

## 💻 Hardware & Environment

- **Google Colab**
  - 2-core Xeon CPU
  - ~13GB RAM
  - GPU: Tesla T4 (when enabled)
- **Local System**
  - **Lenovo ThinkPad**
  - Intel(R) Core(TM) i5-10310U (8 CPUs) @ 1.70GHz
  - 32 GB RAM
  - Windows 10 (64-bit)

---

## 📂 Repository Structure
├── data/ # Raw and processed datasets
├── results/ # Evaluation results and figures
├── notebooks/ # Jupyter/Colab notebooks for experimentation
├── Copy_of_UpdatedSegNet.ipynb Main script for WPBC and WDBC experiments
└── README.md

If you find this work helpful, please cite using the following format:
Sharif, U. (2025). Machine Learning-based Prediction of Breast Cancer Recurrence using WPBC and WDBC Datasets. Unpublished Research.

---

## 🔗 Related Work

- Zuo et al., 2023. 
- Azeroual et al., 2024. 
- Ahmad et al., 2024. 
- Smith et al., 2020.


---

## 📬 Contact

For any queries or collaborations, please contact:  
📧 m.usman.sharif1995@gmail.com  
🏫 Lecturer, Riphah International University, Islamabad  
🔗 GitHub: [@yourgithub](https://github.com/musmansharif)
