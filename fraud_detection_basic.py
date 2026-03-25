import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.01):
    """Generate a synthetic dataset of banking transactions."""
    np.random.seed(42)
    
    # Normal transactions
    n_normal = int(n_samples * (1 - fraud_ratio))
    normal_amount = np.random.normal(loc=100, scale=50, size=n_normal)
    normal_time = np.random.uniform(0, 86400, size=n_normal) # seconds in a day
    normal_loc_dist = np.random.exponential(scale=10, size=n_normal) # distance from home
    
    # Fraud transactions (anomalies)
    n_fraud = int(n_samples * fraud_ratio)
    fraud_amount = np.random.normal(loc=1500, scale=500, size=n_fraud)
    fraud_time = np.random.uniform(0, 86400, size=n_fraud) # uniform throughout day
    fraud_loc_dist = np.random.exponential(scale=500, size=n_fraud) # far from home
    
    # Combine
    amounts = np.concatenate([normal_amount, fraud_amount])
    times = np.concatenate([normal_time, fraud_time])
    loc_dists = np.concatenate([normal_loc_dist, fraud_loc_dist])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Transaction_Amount': amounts,
        'Time_Seconds': times,
        'Distance_From_Home': loc_dists,
        'Is_Fraud': labels
    })
    
    # Ensure no negative amounts or distances
    df['Transaction_Amount'] = df['Transaction_Amount'].clip(lower=1.0)
    df['Distance_From_Home'] = df['Distance_From_Home'].clip(lower=0.0)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def main():
    print("1. Generating Synthetic Banking Data...")
    df = generate_synthetic_data(n_samples=15000, fraud_ratio=0.02)
    print(f"Dataset generated with {len(df)} records.")
    print(f"Fraud distribution:\n{df['Is_Fraud'].value_counts(normalize=True)}\n")
    
    print("2. Preprocessing Data...")
    # Features and Target
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("3. Anomaly Detection (Isolation Forest)...")
    # Isolation Forest is an unsupervised anomaly detection algorithm
    # It attempts to isolate anomalies (frauds) from normal points.
    # The contamination parameter is the expected proportion of outliers.
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    
    # Fit on training data
    iso_forest.fit(X_train_scaled)
    
    # Predict (-1 for outliers, 1 for inliers)
    y_pred_iso = iso_forest.predict(X_test_scaled)
    
    # Convert predictions to 0 for normal, 1 for fraud to match our labels
    y_pred_iso_mapped = np.where(y_pred_iso == -1, 1, 0)
    
    print("\nIsolation Forest Classification Report:")
    print(classification_report(y_test, y_pred_iso_mapped, target_names=['Normal', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_iso_mapped))
    
    print("\n--------------------------------------------------\n")
    
    print("4. Supervised Classification (Random Forest)...")
    # A standard classification model, which performs well with imbalanced data
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    y_pred_clf = clf.predict(X_test_scaled)
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_clf, target_names=['Normal', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_clf))

if __name__ == "__main__":
    main()
