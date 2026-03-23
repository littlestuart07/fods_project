# FODS-DA – Health Risk Data Integration & Modeling

This project performs end-to-end data analysis and machine learning on health-related datasets (diabetes, heart disease, and smoking behavior).

It is organized into two main notebooks:

- `combined_analysis.ipynb`
- `models.ipynb`

---

## 1. Project Overview

### Data cleaning, integration & EDA

Notebook: `combined_analysis.ipynb`

**Goals:**
- Standardize and clean three separate health datasets.
- Merge them into a single, analysis-ready table.
- Handle missing values and engineer additional features.
- Perform basic Exploratory Data Analysis (EDA).

**Input files:**
- `diabetes.csv` – diabetes dataset
- `heart.csv` – heart disease dataset
- `smoking.csv` – smoking behavior dataset

**Key steps:**
- **Column standardization**: convert all column names to lowercase for consistency.
- **Feature selection**:
  - Diabetes: `age`, `glucose`, `bloodpressure`, `bmi`, `outcome`
  - Heart: `age`, `chol`, `trestbps` → renamed to `bloodpressure`, `sex`, `target` → renamed to `heart_disease`
  - Smoking: `age`, `gender`, `smoke`, `amt_weekends`, `amt_weekdays`
- **Merging**:
  - Inner joins on `age` to combine the three datasets into a single `combined` DataFrame.
- **Missing value handling**:
  - Numerical features: median imputation.
  - Categorical features: mode imputation.
- **Feature engineering**:
  - `bmi_category` derived from numeric `bmi`:
    - Underweight / Normal / Overweight / Obese.
- **EDA (examples)**:
  - Diabetes outcome distribution (`outcome`).
  - Heart disease distribution (`heart_disease`).
  - Smoking vs heart disease (countplot).
  - Correlation heatmap for numeric features.

**Output:**
- Cleaned, merged dataset saved as:
  - `combined_cleaned.csv`

---

### Modeling & advanced Random Forest tuning

Notebook: `models.ipynb`

**Goal:** Predict heart disease (`heart_disease`) using multiple machine learning models and compare performance.

**Input:**
- `combined_cleaned.csv` produced in `combined_analysis.ipynb`.

**Preprocessing:**
- Split into:
  - Features `X` (all columns except `heart_disease`)
  - Target `y` (`heart_disease`)
- One-hot encode categorical variables via `pd.get_dummies(drop_first=True)`.
- Train–test split:
  - `test_size=0.2`
  - `random_state=42`
  - `stratify=y` to preserve class balance.
- Imputation + scaling:
  - Median imputation (`SimpleImputer(strategy="median")`) on all numeric columns.
  - Standardization (`StandardScaler`) on features.

**Models trained (STEP 7):**
- Logistic Regression (`LogisticRegression(max_iter=300)`)
- Decision Tree (`DecisionTreeClassifier(max_depth=4, random_state=42)`)
- Random Forest (`RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)`)
- K-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=15)`)
- Gaussian Naive Bayes (`GaussianNB`)

For each model, the notebook prints accuracy on the test set.  
For KNN, a **5-fold cross-validation** accuracy is also integrated using a pipeline (imputation + scaling + KNN) to avoid overfitting and report more realistic performance.

**Typical accuracy ranges (may vary with data):**
- Logistic Regression: ~71%
- Decision Tree: ~75%
- Random Forest (baseline): ~79%
- KNN (k=15, with CV): ~82–88%
- Naive Bayes: ~70%

**Additional evaluation (STEP 8–10):**
- `classification_report` for each model (precision, recall, F1-score).
- Bar plot comparing model accuracies.
- Best model selection based on accuracy.

---

### Advanced Random Forest tuning (STEP 11+)

The project includes an **aggressive hyperparameter search** for Random Forest using `GridSearchCV` with `StratifiedKFold` cross-validation.

**Parameter grid:**

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 6, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}
```

**Setup:**
- Base estimator:

  ```python
  rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
  ```

- Cross-validation:

  ```python
  cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ```

- Grid search:

  ```python
  grid = GridSearchCV(
      estimator=rf_base,
      param_grid=param_grid,
      cv=cv_strategy,
      scoring='accuracy',
      n_jobs=-1,
      verbose=1
  )
  ```

The notebook prints:
- **Best parameters** from the grid search.
- **Best cross-validation accuracy**.
- **Test accuracy before tuning** (baseline Random Forest from STEP 7).
- **Test accuracy after tuning** (best estimator on `X_test`).
- **Percentage improvement** in test accuracy.

**Further evaluation (tuned model):**
- Confusion matrix heatmap.
- ROC curve and AUC.

---

## 2. Environment & Dependencies

This project uses Python 3.x and the scientific Python stack.

**Core libraries:**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter` / `notebook` / `nbconvert`

Install with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook nbconvert
```

---

## 3. How to Run the Notebooks

### Step 1: Clone the repository

```bash
git clone https://github.com/littlestuart07/fods_project.git
cd fods_project
```

### Step 2: Ensure data files are present

The repository already includes:
- `diabetes.csv`
- `heart.csv`
- `smoking.csv`
- `combined_cleaned.csv` (can also be regenerated by running `combined_analysis.ipynb`)

### Step 3: Combined analysis (optional if you want to regenerate `combined_cleaned.csv`)

1. Open `combined_analysis.ipynb` in Jupyter / VS Code / Google Colab.
2. Run all cells from top to bottom.
3. Confirm that `combined_cleaned.csv` is (re)created in the project folder.

### Step 4: Modeling & tuning

1. Open `models.ipynb`.
2. Run all cells from top to bottom.
3. Review:
   - Model accuracy printouts for all 5 models.
   - Accuracy comparison bar plot.
   - Random Forest tuning summary (best params, CV score, before/after test accuracy, % improvement).
   - Confusion matrix and ROC curve for the tuned Random Forest.

---

## 4. Interpretation (High Level)

- Combining diabetes, heart disease, and smoking datasets enables a richer analysis of health risks than any single dataset alone.
- Missing value imputation and feature engineering (e.g., `bmi_category`) help make the data more suitable for modeling.
- Tree-based models and KNN generally perform best, with Random Forest and tuned KNN giving strong, but still realistic, accuracies.
- Cross-validation and a wide hyperparameter grid ensure that performance metrics are robust and not just a result of overfitting to a single train/test split.

---

## 5. License

This repository is for educational and academic purposes.

**This fork / copy:** [https://github.com/littlestuart07/fods_project](https://github.com/littlestuart07/fods_project)

If you reuse or extend this work, please credit this repository. The project builds on ideas and workflows from the original FODS-DA course-style analysis; acknowledge both this repo and any upstream sources you rely on.
