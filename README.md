# 🫀 Heart Disease Prediction Analysis

A machine learning project that applies binary classification to predict the presence of heart disease using the UCI Heart Disease dataset. The project prioritizes **Recall (Sensitivity)** to minimize false negatives — ensuring high-risk patients are flagged for clinical review.

---

## 📌 Project Overview

Cardiovascular disease is one of the leading causes of death globally. This project explores whether machine learning can serve as a scalable, accessible screening tool to identify high-risk individuals before acute events occur.

**Goal:** Binary classification — predict whether a patient has heart disease (1) or not (0)  
**Dataset:** UCI Heart Disease Repository — 920 patient records from 4 institutions  
**Primary Metric:** Recall (Sensitivity) — minimizing dangerous false negatives  

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Name** | Heart Disease Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Records** | 920 patients |
| **Features** | 14 clinical features |
| **Class Balance** | 55.3% with heart disease / 44.7% without |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

---

## 🔬 Methodology

### 1. Data Acquisition & Integration
The dataset was consolidated from four separate CSV sources (Cleveland, Hungary, Switzerland, Long Beach). A key challenge was handling significant missing values across institutions, particularly for features like `ca` and `thal`.

### 2. Exploratory Data Analysis (EDA)
- Generated a **Correlation Heatmap** to identify data quality issues and feature relationships
- `oldpeak` (ST depression) and `exang` (exercise-induced angina) were the **strongest positive predictors**
- `thalach` (maximum heart rate) showed a **strong negative correlation** — higher cardiovascular capacity is a protective indicator
- Surprisingly, **cholesterol ranked only 8th** (correlation: 0.229), challenging the common belief that it is the primary heart disease indicator

### 3. Preprocessing & Feature Engineering

| Step | Approach |
|---|---|
| **Binary Labeling** | Multi-class target (0–4) simplified to binary: healthy (0) vs. disease (1) |
| **Categorical Encoding** | One-Hot Encoding for features like `cp` (chest pain type) and `restecg` |
| **Missing Value Imputation** | Median imputation for low-missingness features; Random Classifier prediction-based imputation for high-missingness features (e.g., `ca`) |

---

## 📊 Models & Results

Three models were evaluated on an **80/20 stratified train-test split**:

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~84% | — | — |
| Decision Tree | ~83% | — | — |
| **Random Forest** | **86%** | **0.81** | **0.91** |

> **Why Recall?** In a medical context, a False Negative (telling a sick person they are healthy) is far more dangerous than a False Positive. The model correctly identified **81% of all disease cases** in the test set.

<!-- Add your charts here once exported from your notebook -->
<!-- ![Model Comparison](images/model_comparison.png) -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->
<!-- ![ROC Curves](images/roc_curves.png) -->

---

## 💡 Key Findings

- **Chest pain type (`cp`)** was the single strongest predictor across both EDA and model-based feature importance — confirming that patient-reported symptoms remain a foundational diagnostic signal
- **Cholesterol underperformed** expectations, ranking 8th — acute indicators like exercise-induced angina and maximum heart rate provide more immediate discriminatory power
- **Random Forest vs. Logistic Regression** presented a strategic trade-off: Random Forest offers slightly higher accuracy, but Logistic Regression provides the transparency essential for medical trust

---

## ⚖️ Ethical Considerations

- **Bias & Equity:** The dataset is historically male-heavy. Without caution, the model may be less accurate for female patients, leading to diagnostic inequity
- **Scope of Use:** This tool is intended as a **decision-support system**, not a replacement for professional medical diagnosis
- **Data Privacy:** Real-world application would require strict adherence to healthcare privacy laws (e.g., HIPAA, PHIPA)

---

## 🚀 Future Directions

- Threshold tuning — lower the decision boundary below 0.5 to further prioritize Recall
- Explore advanced ensemble models (XGBoost, LightGBM) for more complex pattern recognition
- Clinical validation with medical professionals before any real-world deployment

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python heart_disease_model.py
```

---

## 📁 Repository Structure

```
heart-disease-prediction/
│
├── README.md
├── heart_disease_model.py     # Main ML pipeline
├── requirements.txt           # Python dependencies
└── images/                    # Charts and visualizations
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── roc_curves.png
```

---

## 📚 References

- Janosi, A., et al. (1988). *Heart Disease Data Set*. UCI Machine Learning Repository.
- Kiuchi, A., et al. (2024). *Prediction of heart disease severity*. 17th International Joint Conference on Biomedical Engineering Systems.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
