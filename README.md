# 🦠 COVID-19 France: Time Series Analysis and Missing Data Imputation

## 📌 Project Description

This project aims to analyze a time series dataset of COVID-19 cases in France, with a focus on:

- Loading and cleaning the dataset
- Identifying and analyzing missing data
- Applying various imputation techniques
- Evaluating and comparing these techniques using MSE (Mean Squared Error)

The dataset used contains information about confirmed COVID-19 cases and other epidemiological indicators in France.

---

## 📁 Folder Structure

├── COVID/
│ ├── Ressources/
│ |   └──Covid_France.csv
│ ├── Code/
│ |   └──Covid_EDA.ipynb
│ |   └──EDA_Time_Series.py
│ ├── Functions/
│ |   └──def_size_comparison.py
│ ├── requirements.txt
│ └── README.md

---

## ⚙️ Technologies Used

- Python 3.9+
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- SciPy

---

## 🧪 Imputation Techniques Implemented

1. **Moving Average**
2. **Backward Moving Average**
3. **Linear Interpolation**
4. **LOESS/LOWESS Smoothing**
5. **Forward Fill**
6. **Backward Fill**

Each method is evaluated using the **Mean Squared Error (MSE)** metric to compare its effectiveness in estimating missing data.

---

## 📊 Visualization

The notebook provides various time series plots:
- Linear vs Logarithmic scale
- Comparison of all imputation techniques on the same graph
- Histogram of MSE scores

---

## ✅ Results

The evaluation shows that:

- 📈 **Linear Interpolation** and **LOESS/LOWESS** yield the lowest MSE scores
- ❌ **Fill methods** and **simple moving averages** result in significantly higher errors

These results indicate that more sophisticated interpolation methods are better suited for this time series dataset.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/covid-imputation.git
   
