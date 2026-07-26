# FTIR-Based Machine Learning for Recurrent Implantation Failure

Machine learning pipeline for classifying FTIR spectra of Control and Recurrent Implantation Failure (RIF) samples.

## Workflow

Raw FTIR Spectra
        ↓
Savitzky–Golay Smoothing
        ↓
Vector Normalization
        ↓
PCA
        ↓
Model Comparison
        ↓
Random Forest Optimization
        ↓
Nested Cross Validation
        ↓
Feature Importance

## Repository Structure

data/
notebooks/
src/
figures/
results/
models/