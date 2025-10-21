🎓 Student Performance Prediction using Deep Learning (MLP)
📘 Overview

This project analyzes and predicts student academic performance using the Students Performance in Exams dataset from Kaggle. Through EDA, feature engineering, and a Multi-Layer Perceptron (MLP) regression model, it identifies key behavioral and academic factors influencing students’ final grades.

🧠 Project Objectives
Objective
Perform EDA on student demographics, academic habits, and parental involvement.
Preprocess data with encoding, scaling, and feature selection techniques.
Train and evaluate a Deep Learning Regression Model (MLP) to predict final grade (G3).
Visualize trends, correlations, and model performance metrics.
Generate insights to help educators identify at-risk students early.
📊 Dataset Description
Feature Category	Examples
Demographic	sex, age, address, famsize
Social	Medu, Fedu, famsup, Pstatus, goout, health
Academic	studytime, failures, absences, G1, G2, G3

Dataset Summary

Property	Value
Records	1,000+
Features	5
Target Variable	G3 (Final Grade, 0–20 scale)

Source: Kaggle Dataset

🧩 Methodology
1. Data Preprocessing
Step	Description
Duplicate Check	Verified and removed duplicates
Encoding	Applied One-Hot Encoding for categorical variables
Scaling	Standardized numerical attributes using StandardScaler
Outlier Detection	Used IQR method for numeric fields
Data Split	70% train, 15% validation, 15% test
2. Exploratory Data Analysis
Technique	Purpose
Histograms & Boxplots	Analyze grade distribution, gender differences
Heatmap	Identify correlations between grades and behavioral metrics
Scatterplots	Compare G1 vs G3 and studytime vs final grade
Feature Importance	G1, G2, studytime, absences identified as strong predictors
🧮 Deep Learning Model
Architecture
Layer	Neurons	Activation	Purpose
Input	48	ReLU	Accepts preprocessed numeric/categorical features
Hidden Layer 1	128	ReLU	Non-linear feature extraction
Hidden Layer 2	64	ReLU	Hierarchical pattern learning
Dropout	10%	—	Prevents overfitting
Hidden Layer 3	32	ReLU	Higher-order feature abstraction
Output	1	Linear	Predicts continuous grade value
Training Parameters
Parameter	Value	Explanation
Optimizer	Adam	Adaptive learning for efficient convergence
Learning Rate	0.001	Balances speed and stability
Loss Function	MSE	Regression loss
Metrics	MAE, R²	Error magnitude and variance explanation
Epochs	80	Ensures model convergence
Batch Size	32	Stable gradient updates
Validation Split	10%	For generalization monitoring
📈 Results
Metric	Value	Description
RMSE	≈ 2.8	Low prediction error
MAE	≈ 1.9	Small average deviation
R²	≈ 0.78	Explains ~78% of variance

Visual Insights

Plot	Observation
Loss vs Epochs	Smooth downward trend, stable training
Predicted vs Actual	Tight alignment near y = x line
Residual Analysis	Centered near zero → unbiased predictions
🧪 Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib

🚀 How to Run
Step	Action
1	Clone or download the repository
2	Open Jupyter Notebook: jupyter notebook EDA_assignment_2.ipynb
3	Run each cell sequentially: Data loading → Cleaning & visualization → Model building & evaluation
4	Review plots and metrics for insights
📚 Insights & Learnings
Insight
Behavioral metrics and early grades are powerful predictors
Regular attendance and study time directly improve performance
Deep Learning (MLP) handles non-linear educational data better than linear regressors
Feature engineering and scaling significantly affect model quality
🧭 Future Work
Task
Incorporate psychological variables (motivation, stress levels)
Compare with ensemble models (XGBoost, Random Forest)
Build interactive dashboards for real-time student monitoring
Apply Explainable AI (SHAP/LIME) for transparent predictions
Extend to multi-school datasets for improved generalization
👨‍💻 Author

UDHAYA KUMAR K G
B.Tech – Artificial Intelligence and Data Science
KPR Institute of Engineering and Technology

🏁 References
Reference
Students Performance in Exams – Kaggle Dataset

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
Chollet, F. (2018). Deep Learning with Python
Raschka, S., & Mirjalili, V. (2019). Python Machine Learning
McKinney, W. (2017). Python for Data Analysis
Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python
Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment
