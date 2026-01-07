Fuzzy Credit Risk Analyzer and Machine Learning Benchmark
========================================================

Overview
This project implements a credit risk evaluation system that combines a fuzzy logic–based risk scoring model with a machine learning benchmark model (Logistic Regression). The objective is to compare interpretable, rule-based fuzzy inference with statistically trained classification for default risk assessment.

The project includes an interactive Streamlit dashboard that allows users to:
• Evaluate existing customer records
• Manually input customer financial attributes
• Generate fuzzy risk scores
• Compare results with ML-predicted default probability
• View model performance metrics and evaluation summary

-------------------------------------------------------------------------------

Dataset
The application uses the "Default of Credit Card Clients" dataset from UCI / Kaggle.

Dataset link:
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

Download the dataset and place it in the following path:
data/uci_credit_card.csv

Note:
The dataset is not included in this repository to respect licensing and storage constraints.

-------------------------------------------------------------------------------

Key Features
• Fuzzy Inference System with centroid defuzzification
• Logistic Regression ML baseline for performance benchmarking
• Human-readable feature names and interpretable scoring logic
• Real-time risk scoring through an interactive dashboard
• Model performance reporting with ROC-AUC and classification metrics
• Clean and structured UI suitable for academic and demonstration use

-------------------------------------------------------------------------------

Tech Stack
• Python
• NumPy, Pandas
• scikit-learn
• Streamlit
• Custom Fuzzy Logic Rule Engine

-------------------------------------------------------------------------------

Installation
1. Create a virtual environment (recommended)
2. Install project dependencies using:

   pip install -r requirements.txt

-------------------------------------------------------------------------------

Running the Application
To launch the dashboard, run:

   streamlit run app.py

The application will open in a browser window.

-------------------------------------------------------------------------------

Usage Notes
This project is intended for learning, research, and demonstration purposes.  
It should not be used for real-world financial, credit approval, or lending decisions.

-------------------------------------------------------------------------------

Acknowledgement
Dataset credit: UCI / Kaggle – Default of Credit Card Clients Dataset
