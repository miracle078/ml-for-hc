# AKI Prediction with Bayesian Neural Networks
## Clinical Decision Support with Interpretability and Uncertainty Awareness

### Overview
This project implements a Bayesian Neural Network (BNN) for predicting Acute Kidney Injury (AKI) in ICU patients. The model provides uncertainty-aware predictions and includes an adaptive deferral system for cases where the model is uncertain, enabling safer clinical decision support.

### Key Features
- **Bayesian Neural Networks**: Uses Pyro framework for probabilistic modeling
- **Uncertainty Quantification**: Provides prediction confidence intervals
- **Adaptive Deferral System**: Automatically defers uncertain cases to clinical experts
- **Class-Aware Training**: Handles imbalanced datasets with dynamic class weighting
- **Clinical Interpretability**: Uncertainty-aware feature importance analysis
- **Model Calibration**: Ensures predicted probabilities reflect true likelihood

### Requirements
```
torch
pyro-ppl
pandas
numpy
matplotlib
scikit-learn
```

### Installation
```bash
pip install torch pyro-ppl pandas numpy matplotlib scikit-learn
```

### Dataset Requirements
The model expects a CSV file named `aki_data.csv` with the following features:
- **Laboratory values**: creatinine, bicarbonate, chloride, glucose, magnesium, potassium, sodium, urea_nitrogen
- **Blood counts**: hemoglobin, platelet_count, wbc_count
- **Temporal features**: hour_from_icu
- **Target variable**: aki_label (binary: 0/1)
- **Patient identifiers**: stay_id (for temporal splitting)

### Model Architecture
The Bayesian Neural Network consists of:
- Input layer: All clinical features + missingness indicators
- Hidden layer: 32 neurons with ReLU activation
- Output layer: Single neuron with sigmoid activation
- Prior distributions: Normal(0,1) for all weights and biases

### Key Components

#### 1. Data Preprocessing
- **Iterative Imputation**: Robust handling of missing values
- **Missingness Indicators**: Binary flags for missing laboratory values
- **Temporal Context**: Early ICU period indicators
- **Standardization**: Z-score normalization of features

#### 2. Bayesian Inference
- **Variational Inference**: Efficient posterior approximation using SVI
- **Evidence Lower Bound (ELBO)**: Training objective with early stopping
- **Posterior Sampling**: 100 samples for uncertainty quantification

#### 3. Uncertainty-Aware Predictions
- **Predictive Uncertainty**: Standard deviation across posterior samples
- **Adaptive Deferral**: ROC-based threshold for high-uncertainty cases
- **Clinical Metrics**: Sensitivity, specificity, PPV, NPV for non-deferred cases

#### 4. Model Interpretability
- **Calibration Plots**: Assess reliability of predicted probabilities
- **Feature Importance**: Gradient-based analysis across posterior samples
- **Uncertainty Attribution**: Understanding what drives model uncertainty

### Usage

#### Running the Complete Pipeline
```python
# The notebook contains a complete pipeline:
# 1. Data loading and preprocessing
# 2. Feature engineering and temporal splitting
# 3. Bayesian model training
# 4. Uncertainty-aware prediction
# 5. Clinical performance evaluation
# 6. Interpretability analysis
```

#### Key Functions
- `BayesianDNN`: Core model architecture
- `weighted_loss`: Class-aware training objective
- `bayesian_feature_importance`: Uncertainty-aware feature analysis

### Performance Metrics
The model reports:
- **AUC-ROC**: Overall discriminative performance
- **Deferral Rate**: Percentage of cases deferred to experts
- **Clinical Metrics**: Performance on non-deferred cases only
- **Calibration**: Reliability of probability estimates

### Clinical Integration
This model is designed for clinical decision support with:
- **Safety-first approach**: Uncertain cases are deferred to human experts
- **Transparency**: Clear uncertainty quantification for each prediction
- **Interpretability**: Feature importance helps understand model decisions
- **Calibration**: Reliable probability estimates for clinical use

### File Structure
```
AKI Prediction with Uncertainty Awareness.ipynb  # Main notebook
aki_data.csv                                      # Training dataset
README.md                                         # This file
```

### Research Context
This implementation addresses key challenges in clinical AI:
- **Model Uncertainty**: Bayesian approach provides principled uncertainty
- **Class Imbalance**: Weighted loss functions handle skewed datasets
- **Clinical Safety**: Deferral system prevents overconfident predictions
- **Interpretability**: Feature importance analysis for clinical trust

### Future Enhancements
- **Temporal Modeling**: Incorporate time-series features
- **Multi-task Learning**: Predict multiple clinical outcomes
- **Ensemble Methods**: Combine multiple Bayesian models
- **External Validation**: Test on additional hospital datasets

### References
- Pyro Probabilistic Programming: http://pyro.ai/
- Bayesian Deep Learning for Healthcare
- Clinical Decision Support Systems with Uncertainty

### License
This project is for research and educational purposes.
