FODS Project – Health Risk Data Integration & Clustering

This project performs data integration, exploratory data analysis (EDA), and clustering on healthcare datasets related to diabetes, heart disease, and smoking behavior.

The goal is to identify relationships between lifestyle factors such as BMI, glucose level, age, and smoking habits that influence chronic diseases.

1. Project Overview

The project combines multiple healthcare datasets and applies EDA and clustering techniques to discover hidden patterns in health indicators.

Main analysis performed on:

BMI vs Glucose
BMI vs Age
Glucose vs Age

Clustering techniques used:

K-Means Clustering
Agglomerative (Hierarchical) Clustering
Dendrogram Visualization
2. Dataset Files

The following datasets are used:

diabetes.csv
heart.csv
smoking.csv

After preprocessing and integration:

combined_cleaned.csv

Derived datasets:

combined_cleaned_bmi_glucose.csv
combined_cleaned_bmi_age.csv
combined_cleaned_glucose_age.csv
3. Notebooks Used
combined_analysis.ipynb

Performs:

Data cleaning
Data preprocessing
Dataset integration
Feature engineering
Exploratory Data Analysis (EDA)

Output file:

combined_cleaned.csv
models.ipynb

Performs clustering analysis on derived datasets:

K-Means clustering outputs:
combined_cleaned_bmi_glucose_kmeans.csv
combined_cleaned_bmi_age_kmeans.csv
combined_cleaned_glucose_age_kmeans.csv
Agglomerative clustering outputs:
combined_cleaned_bmi_glucose_agglomerative.csv
combined_cleaned_bmi_age_agglomerative.csv
combined_cleaned_glucose_age_agglomerative.csv
Combined cluster results:
combined_cleaned_clusters_all.csv
combined_cleaned_bmi_glucose_clusters_all.csv
combined_cleaned_bmi_age_clusters_all.csv
combined_cleaned_glucose_age_clusters_all.csv
4. Generated Visualizations
K-Means Scatter Plots
kmeans_scatter_bmi_glucose.png
kmeans_scatter_bmi_age.png
kmeans_scatter_glucose_age.png
Agglomerative Scatter Plots
agglomerative_scatter_bmi_glucose.png
agglomerative_scatter_bmi_age.png
agglomerative_scatter_glucose_age.png
Dendrograms
agglomerative_dendrogram_bmi_glucose.png
agglomerative_dendrogram_bmi_age.png
agglomerative_dendrogram_glucose_age.png
5. Python Scripts
kmeans_cluster.py – performs K-Means clustering
kmeans_cluster_plot.py – generates K-Means cluster visualization plots
6. Libraries Used
Python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter notebook

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn scipy notebook
7. How to Run the Project
Step 1: Clone repository
git clone https://github.com/littlestuart07/fods_project.git
cd fods_project
Step 2: Run data preprocessing notebook

Open:

combined_analysis.ipynb

Run all cells to generate:

combined_cleaned.csv

Step 3: Run clustering notebook

Open:

models.ipynb

Run all cells to generate:

cluster CSV files
scatter plots
dendrogram images
8. Key Learning Outcomes
Data cleaning and preprocessing of healthcare datasets
Dataset integration from multiple sources
Exploratory Data Analysis (EDA)
Feature engineering
K-Means clustering implementation
Agglomerative hierarchical clustering
Dendrogram interpretation
Visualization of health indicator relationships
9. Project Repository

GitHub Repository:

https://github.com/littlestuart07/fods_project

10. Academic Purpose

This project is developed for Foundation of Data Science (FODS) coursework.

It demonstrates how data science techniques can be used to analyze healthcare datasets and discover patterns related to chronic disease risk factors.