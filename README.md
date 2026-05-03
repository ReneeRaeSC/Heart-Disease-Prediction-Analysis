# Heart Disease Prediction Analysis

A machine learning project that applies binary classification to predict the presence of heart disease using the UCI Heart Disease dataset. The project prioritizes **Recall (Sensitivity)** to minimize false negatives — ensuring high-risk patients are flagged for clinical review.

---

## Project Overview

Cardiovascular disease is one of the leading causes of death globally. This project explores whether machine learning can serve as a scalable, accessible screening tool to identify high-risk individuals before acute events occur.

| | |
|---|---|
| **Goal** | Binary classification — predict whether a patient has heart disease (1) or not (0) |
| **Dataset** | UCI Heart Disease Repository — 920 patient records from 4 institutions |
| **Models** | Logistic Regression, Decision Tree, Random Forest |
| **Primary Metric** | Recall (Sensitivity) — minimizing dangerous false negatives |

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Heart Disease Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Records** | 920 patients (Cleveland, Hungary, Switzerland, Long Beach) |
| **Features** | 14 clinical features |
| **Class Balance** | 55.3% with heart disease / 44.7% without |

<details>
<summary> Feature Dictionary</summary>

| Variable | Name | Description |
|---|---|---|
| `age` | Age | Age in years |
| `sex` | Sex | 1 = male; 0 = female |
| `cp` | Chest Pain Type | 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic |
| `trestbps` | Resting Blood Pressure | In mm Hg on admission to hospital |
| `chol` | Serum Cholesterol | In mg/dl |
| `fbs` | Fasting Blood Sugar | > 120 mg/dl (1 = true; 0 = false) |
| `restecg` | Resting ECG | 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy |
| `thalach` | Max Heart Rate | Maximum heart rate achieved |
| `exang` | Exercise Angina | Exercise induced angina (1 = yes; 0 = no) |
| `oldpeak` | ST Depression | ST depression induced by exercise relative to rest |
| `slope` | ST Slope | Slope of the peak exercise ST segment |
| `ca` | Major Vessels | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia | 3 = normal; 6 = fixed defect; 7 = reversible defect |
| `target` | Diagnosis | Disease status (binary: 0 = no disease, 1 = disease) |

</details>

---

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

---

## Methodology

### 1. Data Acquisition & Integration
The dataset was consolidated from four separate CSV sources downloaded directly from the UCI repository. A key challenge was handling significant missing values across institutions, particularly for `ca` (66% missing) and `thal` (53% missing).

### 2. Exploratory Data Analysis (EDA)

Key findings from correlation analysis:

| Feature | Correlation with Heart Disease | Direction |
|---|---|---|
| `cp` (Chest Pain Type) | **0.472** | Positive |
| `exang` (Exercise Angina) | 0.434 | Positive |
| `thalach` (Max Heart Rate) | 0.382 | Negative ✦ |
| `oldpeak` (ST Depression) | 0.366 | Positive |
| `thal` (Thalassemia) | 0.312 | Positive |
| `sex` | 0.307 | Positive |
| `chol` (Cholesterol) | 0.229 | Positive |

> ✦ Higher cardiovascular capacity is a **protective** indicator — patients with higher max heart rates were less likely to have heart disease.
>
> **Notable finding:** Cholesterol ranked only 8th despite being the most widely cited heart disease risk factor.

### 3. Preprocessing & Feature Engineering

| Step | Approach |
|---|---|
| **Binary Labeling** | Multi-class target (0–4) simplified to binary: healthy (0) vs. disease (1) |
| **Categorical Encoding** | One-Hot Encoding (`drop_first=True`) for features like `cp` and `restecg` |
| **Median Imputation** | Applied to low-missingness features (`trestbps`, `chol`, `fbs`, etc.) — resistant to extreme values |
| **RF-Based Imputation** | Applied to high-missingness features (`ca` 66%, `thal` 53%, `slope`) — predicts missing values using correlations from complete records |
| **Train/Test Split** | 80/20 stratified split (preserves class ratio in both sets) |
| **Feature Scaling** | StandardScaler applied for Logistic Regression |

---

## Results

### Single Split Performance (80/20 Test Set)

| Model | Recall (Disease) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.86 | 0.9035 |
| Decision Tree | 0.83 | 0.8535 |
| **Random Forest** | **0.88** | **0.9164** |

> **Why Recall?** A False Negative (telling a sick patient they are healthy) is far more dangerous than a False Positive. Random Forest correctly identified **88% of all actual heart disease cases**.

### Cross-Validation (5-Fold Stratified)

| Model | CV ROC-AUC | Std Dev |
|---|---|---|
| Logistic Regression | 0.8785 | ±0.0242 |
| Decision Tree | 0.7884 | ±0.0365 |
| **Random Forest** | **0.8814** | ±0.0275 |

> Cross-validation is the more reliable estimate. Random Forest leads (0.8814 vs 0.8785), but the gap is only **0.0029** — a very narrow margin. Logistic Regression is more stable (lower variance) and fully explainable, making it a strong clinical alternative.

<!-- Add your exported charts here -->
<!-- ![Model Performance Comparison](images/model_comparison.png) -->
<!-- ![Confusion Matrices](images/confusion_matrix.png) -->
<!-- ![ROC Curves](images/roc_curves.png) -->

---

## Key Findings

**1. Chest pain type is the strongest predictor.**
With a correlation of 0.472, `cp` was the top-ranked feature in both EDA and all three model feature importance rankings — confirming it as the most reliable clinical signal in this dataset.

**2. Cholesterol underperformed expectations.**
Despite being the most widely cited risk factor, cholesterol ranked 8th (correlation: 0.229). Exercise-induced angina and max heart rate proved far more discriminating.

**3. Random Forest vs. Logistic Regression is essentially a tie.**
Random Forest leads by only 0.0029 in cross-validated AUC. Logistic Regression is more stable (±0.0242 vs ±0.0275) and fully explainable — an important consideration in clinical settings where doctors need to understand and trust a prediction.

**4. Decision Tree is the weakest model** under both evaluation methods and is not recommended for deployment.

---

## Ethical Considerations

- **Bias & Equity:** The dataset is historically male-heavy. Without caution, the model may be less accurate for female patients, leading to diagnostic inequity
- **Scope of Use:** This tool is intended as a **decision-support system**, not a replacement for professional medical diagnosis
- **Data Privacy:** Real-world application would require strict adherence to healthcare privacy laws (e.g., HIPAA, PHIPA)

---

## Future Directions

1. **Hyperparameter tuning** — Use GridSearchCV to find optimal settings and verify whether the LR vs RF tie holds after tuning
2. **Threshold tuning** — Lower the decision threshold below 0.5 to further prioritize Recall and reduce missed diagnoses
3. **Advanced models** — Test XGBoost or LightGBM for more complex pattern recognition
4. **Clinical validation** — Work with medical professionals to validate predictions before any real-world deployment

---

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Heart_Disease_Prediction_Analysis.ipynb
```

---

## Repository Structure

```
heart-disease-prediction/
│
├── README.md
├── Heart_Disease_Prediction_Analysis.ipynb   # Full analysis notebook
├── requirements.txt                          # Python dependencies
└── images/                                   # Exported chart images
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── roc_curves.png
```

---

## References

- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease Data Set*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- Kiuchi, A., Fujita, T., & Yamana, H. (2024). *Prediction of heart disease severity using hierarchically-structured machine-learning models*. 17th International Joint Conference on Biomedical Engineering Systems and Technologies. https://doi.org/10.5220/0012436700003657
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
