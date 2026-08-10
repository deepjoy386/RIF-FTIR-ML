# FTIR-Based Machine Learning for Recurrent Implantation Failure Classification

A reproducible machine learning pipeline for classifying **Recurrent Implantation Failure (RIF)** and **Control** samples using **Fourier Transform Infrared (FTIR) spectroscopy**.

This repository contains the complete workflow from spectral preprocessing to model evaluation and interpretation, developed as part of a research project.

---

## Project Overview

Recurrent Implantation Failure (RIF) is a major challenge in assisted reproductive technologies. This project investigates whether FTIR spectra can be used to distinguish between **RIF** and **Control** samples using supervised machine learning.

The pipeline includes:

- FTIR spectral preprocessing
- Exploratory data analysis
- Pre-specified univariate spectral band-area comparisons
- Machine learning model comparison
- Random Forest hyperparameter tuning
- Nested Cross Validation
- Model evaluation
- Publication-quality visualizations

---

## Dataset

| Property | Value |
|----------|------:|
| Total Samples | 72 |
| Control Samples | 36 |
| RIF Samples | 36 |
| Spectral Variables | 484 |
| Class Distribution | Balanced |

---

## Workflow

```text
Raw FTIR Spectra
        │
        ▼
Savitzky–Golay Smoothing
        │
        ▼
Vector Normalization
        │
        ▼
Univariate Spectral Band-Area Analysis
        │
        ▼
Machine Learning Model Comparison
        │
        ▼
Random Forest Hyperparameter Optimization
        │
        ▼
Nested Cross Validation
        │
        ▼
```

---

## Repository Structure

```text
RIF-FTIR-ML/

├── data/
│   ├── RIF_FTIR.csv
│   └── processed_spectra.csv
│
├── figures/
│   ├── Figure1_RawSpectra.png
│   ├── Figure2_MeanSpectra.png
│   ├── Figure3_UnivariateBandAreas.png
│   ├── Figure4_ModelComparison.png
│   ├── Figure5_ROC.png
│   ├── Figure6_ConfusionMatrix.png
│   └── Figure7_SpectralInterpretation.png
│
├── models/
│   └── random_forest_best.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 04_machine_learning.ipynb
│   ├── 05_random_forest_tuning.ipynb
│   ├── 06_final_evaluation.ipynb
│   └── 07_feature_importance.ipynb
│
├── results/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── permutation_importance.csv
│   └── top_regions.csv
│
├── src/
│   ├── preprocessing.py
│   ├── transformers.py
│   ├── pipelines.py
│   ├── models.py
│   ├── evaluation.py
│   ├── interpretation.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Models

The following supervised learning algorithms are evaluated:

- Linear Discriminant Analysis (LDA)
- Partial Least Squares Discriminant Analysis (PLS-DA)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Model selection was performed using repeated stratified cross-validation, followed by hyperparameter optimization for the Random Forest classifier.

---

## Preprocessing

The FTIR spectra were preprocessed using:

- Savitzky–Golay smoothing
- Vector (L2) normalization

These preprocessing steps were incorporated into the machine learning pipeline to prevent data leakage during cross-validation.

The mean-spectrum figure is calculated from the saved **processed spectra** (smoothed and vector-normalized), rather than from raw spectra.

---

## Evaluation

The optimized Random Forest model was evaluated using **nested cross-validation**.

Performance metrics include:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Project Outputs

The repository contains:

- Raw and processed FTIR spectra
- Univariate spectral band-area comparison
- Model comparison
- ROC curve
- Confusion matrix
- Publication-quality figures
- Trained Random Forest model

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/deepjoy386/RIF-FTIR-ML.git

cd RIF-FTIR-ML
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the notebooks in the following order:

1. `01_data_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `04_machine_learning.ipynb`
4. `05_random_forest_tuning.ipynb`
5. `06_final_evaluation.ipynb`

To reproduce the feedback updates in one step, run:

```bash
python scripts/generate_feedback_outputs.py
```

This regenerates `data/processed_spectra.csv`, the processed mean-spectrum plot,
the spectral-projection figure, and two data-driven grouped model-comparison
plots. To generate the univariate band-area comparison figure, run
`python scripts/plot_univariate_ftir.py`.

To reproduce the nested-CV comparison and the fold-specific optimized
hyperparameters for every model, run:

```bash
python scripts/run_nested_cv_all_models.py
```

This creates `figures/Figure4_NestedCVModelComparison.png` and
`results/optimized_hyperparameters.csv`. The hyperparameter CSV reports the
settings selected by the inner cross-validation for each outer fold, together
with its corresponding inner-CV ROC-AUC.

---

## Results

The workflow produces:

- Exploratory FTIR visualizations
- Univariate spectral band-area comparisons
- Performance comparison of multiple classifiers
- Optimized Random Forest model
- ROC curve
- Confusion matrix

---

## Future Work

Potential extensions include:

- External validation using an independent dataset
- Deep learning models for spectral classification
- Explainable AI techniques (e.g., SHAP)
- Integration with clinical metadata
- Deployment as a reproducible analysis pipeline

---

## Author

**Deepjoy Das**

B.Tech, Agricultural and Food Engineering  
Indian Institute of Technology Kharagpur

---

## License

This project is intended for academic and research purposes. Please consult the repository license for usage terms.
