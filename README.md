# End-to-End California Housing ML

## 📌 Overview

This repository contains an **end-to-end machine learning project** for predicting **housing prices in California**, following best practices presented in *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

The project demonstrates a complete real-world ML workflow, starting from raw data exploration and preprocessing, through model training and evaluation, to building reusable pipelines.

---

## 🎯 Project Objectives

* Understand the California Housing dataset
* Perform Exploratory Data Analysis (EDA)
* Apply proper data cleaning and feature engineering
* Build preprocessing and training pipelines
* Train and evaluate multiple regression models
* Tune hyperparameters to improve performance

---

## 🗂️ Project Structure

```
end-to-end-california-housing-ml/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and prepared data
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── pipelines.py
│   ├── train.py
│   └── evaluate.py
│
├── models/                  # Saved trained models
├── reports/                 # Figures and evaluation results
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* **Dataset**: California Housing Dataset
* **Source**: StatLib / Scikit-learn
* **Target**: `median_house_value`
* **Features include**:

  * Median income
  * Housing median age
  * Average rooms
  * Population
  * Latitude & longitude

---

## 🧪 Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* (Optional) Gradient Boosting Regressor

Model performance is evaluated using:

* RMSE (Root Mean Squared Error)
* Cross-validation

---

## ⚙️ Technologies & Tools

* Python
* NumPy
* Pandas
* Matplotlib & Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/abanoub-refaat/california-housing-end-to-end-ml.git
cd california-housing-end-to-end-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebooks in order inside the `notebooks/` directory.

---

## 📚 Learning Reference

This project closely follows concepts from:

> *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron

It is intended for **learning, practice, and portfolio demonstration**.

---

## 👤 Author

**Abanoub Refaat**
Frontend & Machine Learning Enthusiast

* GitHub: [https://github.com/abanoub-refaat](https://github.com/abanoub-refaat)
* LinkedIn: [https://www.linkedin.com/in/abanoubrefaat/](https://www.linkedin.com/in/abanoubrefaat/)

---

## ⭐ Acknowledgments

* Aurélien Géron for the HOML book
* Scikit-learn community

If you find this project useful, feel free to ⭐ the repository!
