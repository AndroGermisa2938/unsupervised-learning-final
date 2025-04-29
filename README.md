# Anomaly Detection in Financial Transactions  
Using Unsupervised Learning on the PaySim Dataset

## Overview
This project applies unsupervised machine learning algorithms to detect fraudulent transactions in financial data using the [PaySim simulation dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).

Due to the rarity of fraud and the lack of labeled training data, we use **anomaly detection techniques** to identify unusual patterns in transaction behavior.

---

## Dataset Summary

- Over 6 million simulated mobile money transactions
- Multiple transaction types (TRANSFER, CASH_OUT, etc.)
- Only ~0.13% of transactions are fraudulent
- Severe class imbalance → suitable for anomaly detection

---

## Models Implemented

We used the following unsupervised anomaly detection techniques:

1. **Isolation Forest**
   - Tree-based model that isolates anomalies through random partitioning
   - Efficient and performs well on high-dimensional data

2. **Local Outlier Factor (LOF)**
   - Detects anomalies by comparing local density of data points
   - High recall, more sensitive to outliers

3. **One-Class SVM**
   - Learns a boundary around normal data
   - Flags deviations as anomalies

---

## Key Techniques Used

- Feature scaling and encoding
- Contamination tuning (fraction of expected anomalies)
- Precision-Recall evaluation
- Visualizations of anomaly scores and distributions

---

## Setup Instructions

1. Clone the repository:

```
git clone https://github.com/AndroGermisa2938/unsupervised-learning-final
cd anomaly-detection
```

2. Download the data, and extract it under anomaly-detection/data

3. Run the requirements.txt

```
pip install -r requirements.txt
```
