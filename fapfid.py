import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

def FAPFID_algorithm(data, protected_feature, privileged_group, unprivileged_group, base_classifier, num_clusters):
    """
    FAPFID approace.
    
    Parameters needed:
    - data: Input pandas DataFrame containing features and target.
    - protected_feature: Name of the protected feature (e.g., 'gender').
    - privileged_group: Value indicating the privileged group (e.g., 'male').
    - unprivileged_group: Value indicating the unprivileged group (e.g., 'female').
    - base_classifier: A scikit-learn classifier (e.g., DecisionTreeClassifier).
    - num_clusters: Number of clusters for the clustering algorithm.

    Returns:
    - ensemble_model: Trained ensemble model.
    - metrics: Dictionary containing accuracy, balanced accuracy, and fairness scores.
    """

    # Step 1: Split data into K clusters using KMeans clustering
    features = data.drop(columns=['target'])
    target = data['target']
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)

    # Step 2: Balance imbalanced clusters based on the privileged-to-unprivileged ratio
    balanced_clusters = []
    smote = SMOTE(random_state=42)

    for cluster_id in range(num_clusters):
        # Extract the current cluster => contains all the rows from the original DataFrame data that belong to cluster_id.
        cluster_data = data[data['cluster'] == cluster_id]
        privileged = cluster_data[cluster_data[protected_feature] == privileged_group]
        unprivileged = cluster_data[cluster_data[protected_feature] == unprivileged_group]

        # Compute the balance ratio (rp)
        if len(unprivileged) == 0:  # Avoid division by zero
            rp = float('inf')  # Fully privileged cluster
        elif len(privileged) == 0:  # Fully unprivileged cluster
            rp = 0
        else:
            rp = len(privileged) / len(unprivileged)

        # Check if the cluster is imbalanced
        if np.isclose(rp, 1.0):  # If balanced (rp ≈ 1), add directly to the final set
            balanced_clusters.append(cluster_data)
        else:  # If imbalanced, apply SMOTE to the entire cluster
            X = cluster_data.drop(columns=['target', 'cluster'])
            y = cluster_data['target']
            
            # Apply SMOTE to balance the cluster
            X_bal, y_bal = smote.fit_resample(X, y)
            
            # Reconstruct the oversampled cluster
            cluster_balanced = pd.DataFrame(X_bal, columns=X.columns)
            cluster_balanced['target'] = y_bal
            cluster_balanced['cluster'] = cluster_id
            
            balanced_clusters.append(cluster_balanced)

    # Step 3: Create a balanced dataset X' from all clusters
    balanced_data = pd.concat(balanced_clusters, axis=0)

    # Step 4: Create bags for bagging
    num_bags = 2 * num_clusters + 1
    ensemble_models = []
    for i in range(num_bags):
        bootstrap_sample = balanced_data.sample(frac=1, replace=True, random_state=42 + i)
        X_train = bootstrap_sample.drop(columns=['target', 'cluster'])
        y_train = bootstrap_sample['target']
        
        # Train base classifier
        model = clone(base_classifier)
        model.fit(X_train, y_train)
        ensemble_models.append(model)

    # Step 5: Ensemble model by majority voting
    def ensemble_predict(X):
        predictions = np.array([model.predict(X) for model in ensemble_models])
        # Majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    # Final ensemble model
    ensemble_model = BaggingClassifier(base_estimator=base_classifier, n_estimators=num_bags, random_state=42)
    ensemble_model.fit(balanced_data.drop(columns=['target', 'cluster']), balanced_data['target'])

    # Performance metrics
    X_test = data.drop(columns=['target', 'cluster'])
    y_test = data['target']
    y_pred = ensemble_predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Calculate Equalized Odds fairness metrics using fairlearn
    # sensitive_features: the protected feature column in the dataset (e.g., gender)
    sensitive_features = data[protected_feature]
    
    # Equalized odds difference and ratio
    eq_odds_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eq_odds_ratio = equalized_odds_ratio(y_test, y_pred, sensitive_features=sensitive_features)
    
    fairness_metrics = {
        'equalized_odds_difference': eq_odds_diff,
        'equalized_odds_ratio': eq_odds_ratio
    }
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'fairness': fairness_metrics
    }
    
    return ensemble_model, metrics
