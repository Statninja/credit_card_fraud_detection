# 💳 Credit Card Fraud Detection Project 🔍

This project focuses on detecting fraudulent credit card transactions using data analysis, visualization, and advanced machine learning techniques. We explored the data, visualized important patterns, and built predictive models to accurately classify fraud.

---

## 🛠️ Project Overview

This project utilizes a dataset of credit card transactions, identifying key characteristics of fraudulent transactions and developing a machine learning model to distinguish them from legitimate ones.

---

## 📝 Steps Taken

1. **🚀 Import Libraries**  
   - Loaded libraries for data manipulation, visualization, and machine learning: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `sklearn`, `imbalanced-learn`.

2. **📂 Load and Inspect Data**  
   - Loaded the CSV file and inspected basic data info, missing values, and data types.

3. **🔍 Exploratory Data Analysis (EDA)**  
   - Examined class distribution of fraud vs. legitimate transactions.
   - Reviewed feature correlations and variable distributions.

4. **📊 Data Visualization**  
   - Developed visuals to uncover patterns, distributions, and relationships in the data (see full list below 👇).

5. **🔄 Data Preprocessing**  
   - Standardized features to improve consistency.
   - Applied SMOTE and other resampling techniques to handle data imbalance.

6. **🧠 Machine Learning Models**  
   - Split data into training and testing sets.
   - Trained and tuned multiple models, including Logistic Regression, Random Forest, Gradient Boosting, SVM, and Neural Networks.
   - Employed cross-validation and grid search to enhance model performance.

7. **📈 Model Evaluation & Visualization**  
   - Evaluated models using accuracy, AUC-ROC, F1-score, precision, and recall.
   - Visualized model outputs, including confusion matrices, ROC curves, and Precision-Recall curves.

8. **📜 Conclusion**  
   - Documented key insights and provided recommendations for deployment and further analysis.

---

## 📊 Visuals and Analysis Included

### 1. **📈 Fraud vs. Legitimate Transaction Distribution**  
   - Bar plot highlighting the significant class imbalance.

### 2. **📉 Transaction Amounts Scatter Plot**  
   - Scatter plot of transaction amounts over time to identify trends or unusual spikes.

### 3. **📊 Histogram Amounts for Fraudulent vs. Legitimate**  
   - Box plots to compare transaction amounts, exposing outliers in fraud cases.

### 4. **🌐 Correlation Heatmap**  
   - Heatmap visualizing feature correlations, helping to identify variables related to fraud.(Not included as the numbers were not accurate)

### 5. **💹 PCA for Dimensionality Reduction**  
   - PCA visualization to simplify feature space, helping to distinguish fraud from legitimate transactions.

### 6. **📊 Model Confusion Matrices**  
   - Confusion matrices for each model, showing how well each model performs in fraud vs. legitimate classifications.

### 7. **📈 AUC-ROC Curve for Model Comparison**  
   - ROC curves for Logistic Regression, Random Forest, Gradient Boosting, SVM, and Neural Network models, displaying model quality.

### 8. **📉 Precision-Recall Curves**  
   - Precision-Recall curves highlighting trade-offs between precision and recall for each model.

### 9. **🤖 SHAP Summary Plot**  
   - Used SHAP (SHapley Additive exPlanations) to visualize the most impactful features on model predictions.

### 10. **📉 Feature Importance Plot for Tree Models**  
   - Feature importance visualization from the Random Forest and Gradient Boosting models, showing which variables drive the classification.

---

## ⚙️ Machine Learning Models and Techniques

1. **🔍 Logistic Regression**
   - Provided baseline metrics and probability-based output.

2. **🌲 Random Forest**  
   - Ensemble model capturing complex patterns and interactions.

3. **🚀 Gradient Boosting**  
   - Boosted trees for improved performance in classifying minority fraud cases.

4. **🕹️ Support Vector Machine (SVM)**  
   - Classified using hyperplanes with an emphasis on maximizing class separation.

5. **🧠 Neural Network**  
   - Deep learning model for complex fraud pattern recognition.

---

## 📈 Model Evaluation Metrics

- **Confusion Matrix**  
   - Shows true positive, true negative, false positive, and false negative counts for fraud detection.
- **AUC-ROC Curve**  
   - Compared models using AUC to understand overall classification power.
- **Precision-Recall Analysis**  
   - Emphasized precision (minimizing false positives) and recall (capturing fraud cases) trade-offs.

---

## 📜 Summary

This project demonstrates a workflow for detecting credit card fraud using data visualization and machine learning. Handling class imbalance and prioritizing accurate fraud detection metrics were crucial, and our analysis identified effective models and techniques to reduce fraud risk.

---

## 🏃‍♂️ Run It Yourself

1. **Download the notebook and dataset**.
2. **Install required libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn shap

