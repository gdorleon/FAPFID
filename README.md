# FAPFID: A Fairness-Aware Approach for Clustering and Ensemble Learning

This repository contains the  implementation of the **FAPFID algorithm**, a fairness-aware method that balances clusters based on demographic parity and trains an ensemble model for classification tasks. The approach ensures fairness by identifying and oversampling imbalanced clusters while maintaining group fairness metrics.

The paper was first published here:  
**Fapfid: A fairness-aware approach for protected features and imbalanced data**  
G Dorleon, I Megdiche, N Bricon-Souf, O Teste - *Transactions on Large-Scale Data-and Knowledge …*, 2023
https://link.springer.com/chapter/10.1007/978-3-662-66863-4_5

---

## Overview

The FAPFID algorithm follows these steps:

1. **Clustering**:
   - The dataset is divided into `K` clusters using a KMeans.

2. **Balanced Clusters**:
   - Each cluster is checked for imbalance based on the ratio of privileged to unprivileged instances (`rp`).
   - Clusters with `rp ≠ 1` (imbalanced clusters) are oversampled using **SMOTE** to achieve balance.

3. **Bagging Ensemble**:
   - From the final set of balanced clusters, bootstrap samples are created to train multiple base classifiers.
   - An ensemble model is constructed through majority voting.

4. **Fairness Evaluation**:
   - The final model is evaluated for fairness using the **Equalized Odds** metric, along with accuracy and balanced accuracy.

---

## Features

- **Fairness-Aware Oversampling**: Balances clusters based on demographic parity without globally altering class distributions.
- **Customizable Clustering**: Supports clustering algorithms (default: KMeans).
- **SMOTE-Based Balancing**: Balances only imbalanced clusters to mitigate group bias.
- **Bagging Ensemble Learning**: Builds robust ensemble models by training classifiers on bootstrap samples.

---

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

```bash
pip install requirement.txt
````
## Usage

### 1. Input Data Requirements

The input dataset should be a CSV file containing the following:

* **Features**: Columns with feature data for clustering and classification.
* **Target**: A column named `target` representing the classification target.
* **Protected Feature**: A column representing the protected attribute (e.g., `gender`, `race`).

The dataset should also contain:

* **Privileged Group**: Value(s) representing the privileged group (e.g., `male`, `white`).
* **Unprivileged Group**: Value(s) representing the unprivileged group (e.g., `female`, `black`).

### 2. Example Code

```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Define parameters
protected_feature = 'gender'  # Name of the protected feature
privileged_group = 'male'     # Privileged group value
unprivileged_group = 'female' # Unprivileged group value
base_classifier = DecisionTreeClassifier(random_state=42)  # Base classifier
num_clusters = 5  # Number of clusters

# Run the FAPFID algorithm
ensemble_model, metrics = FAPFID_algorithm(
    data=data,
    protected_feature=protected_feature,
    privileged_group=privileged_group,
    unprivileged_group=unprivileged_group,
    base_classifier=base_classifier,
    num_clusters=num_clusters
)

# Output performance metrics
print("Accuracy:", metrics['accuracy'])
print("Balanced Accuracy:", metrics['balanced_accuracy'])
print("Fairness (Equalized Odds):", metrics['fairness'])

