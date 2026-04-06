# Ship Engine Anomaly Detection
### Predictive Maintenance | Cambridge Data Science Programme

An anomaly detection system for a shipping company's fleet, built to 
predict engine failures before they occur, reducing downtime, safety 
risks and fuel consumption.

---

## Problem

A shipping fleet needed to identify vessels displaying early signs of 
engine anomalies across 6 sensor features: engine RPM, lubrication oil 
pressure, fuel pressure, coolant pressure, lubrication oil temperature 
and coolant temperature.

---

## Approach

Three anomaly detection methods were applied and compared, combining 
univariate statistical and multivariate machine learning techniques:

| Method | Type | Anomalies Detected |
|--------|------|--------------------|
| IQR (≥2 features) | Statistical · Univariate | 422 ships (2.16%) |
| One-Class SVM (ν=0.01, γ=0.2) | ML · Multivariate | 202 ships (1.03%) |
| Isolation Forest (contamination=0.01) | ML · Multivariate | 196 ships (1.00%) |

---

## Key Findings

- **Isolation Forest** identified as the optimal model, computationally 
  efficient, not sensitive to feature scaling, suitable for large datasets
- Strong agreement between SVM and Isolation Forest 
  validates both models' reliability
- **196 ships** flagged for immediate inspection and maintenance
- PCA visualisation confirmed anomalies cluster distinctly at the 
  periphery of normal operating data

---

## Recommendation

Isolation Forest should be deployed as the primary anomaly detection 
model. Ships flagged at contamination=0.01 represent the highest 
confidence anomaly cases requiring priority maintenance intervention.

---

## Methods
```python
# Core libraries
sklearn.svm          # OneClassSVM
sklearn.ensemble     # IsolationForest
sklearn.decomposition # PCA
scipy.stats          # IQR calculation
```

---

## Author

**Israel Olujobi**  
Cambridge Data Science & AI Programme (December 2025)
