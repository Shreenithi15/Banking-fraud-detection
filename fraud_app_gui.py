import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ Banking Transaction Fraud Detection")
        self.root.geometry("600x600")
        self.root.configure(padx=20, pady=20)
        
        self.df = None
        self.scaler = None
        self.iso_forest = None
        self.clf = None
        
        self.setup_ui()
        self.root.after(100, self.train_models)

    def setup_ui(self):
        # Title
        title_label = ttk.Label(self.root, text="Banking Transaction Fraud Detection", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Initializing models... Please wait.")
        status_label = ttk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 10, "italic"), foreground="grey")
        status_label.pack(pady=(0, 20))
        
        # Frame for Input
        input_frame = ttk.LabelFrame(self.root, text="New Transaction Details", padding=(10, 10))
        input_frame.pack(fill="x", pady=10)
        
        # Amount
        ttk.Label(input_frame, text="Transaction Amount ($):").grid(row=0, column=0, sticky="w", pady=5)
        self.amount_var = tk.DoubleVar(value=150.0)
        ttk.Entry(input_frame, textvariable=self.amount_var).grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        
        # Time
        ttk.Label(input_frame, text="Time of Transaction (Hour of Day 0-24):").grid(row=1, column=0, sticky="w", pady=5)
        self.time_var = tk.DoubleVar(value=12.0)
        ttk.Entry(input_frame, textvariable=self.time_var).grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        
        # Distance
        ttk.Label(input_frame, text="Distance From Home (Miles):").grid(row=2, column=0, sticky="w", pady=5)
        self.distance_var = tk.DoubleVar(value=25.0)
        ttk.Entry(input_frame, textvariable=self.distance_var).grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        
        input_frame.columnconfigure(1, weight=1)
        
        # Predict Button
        self.predict_btn = ttk.Button(self.root, text="Predict Fraud", command=self.predict, state="disabled")
        self.predict_btn.pack(pady=20, fill="x")
        
        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding=(10, 10))
        results_frame.pack(fill="both", expand=True, pady=10)
        
        # RF Result
        ttk.Label(results_frame, text="Random Forest (Supervised):", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.rf_res_var = tk.StringVar(value="Waiting...")
        self.rf_res_label = ttk.Label(results_frame, textvariable=self.rf_res_var, font=("Helvetica", 10))
        self.rf_res_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # ISO Result
        ttk.Label(results_frame, text="Isolation Forest (Anomaly):", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.iso_res_var = tk.StringVar(value="Waiting...")
        self.iso_res_label = ttk.Label(results_frame, textvariable=self.iso_res_var, font=("Helvetica", 10))
        self.iso_res_label.grid(row=1, column=1, sticky="w", padx=10, pady=5)

    def generate_data(self, n_samples=15000, fraud_ratio=0.02):
        np.random.seed(42)
        n_normal = int(n_samples * (1 - fraud_ratio))
        normal_amount = np.random.normal(loc=100, scale=50, size=n_normal)
        normal_time = np.random.uniform(0, 86400, size=n_normal)
        normal_loc_dist = np.random.exponential(scale=10, size=n_normal)
        
        n_fraud = int(n_samples * fraud_ratio)
        fraud_amount = np.random.normal(loc=1500, scale=500, size=n_fraud)
        fraud_time = np.random.uniform(0, 86400, size=n_fraud)
        fraud_loc_dist = np.random.exponential(scale=500, size=n_fraud)
        
        amounts = np.concatenate([normal_amount, fraud_amount])
        times = np.concatenate([normal_time, fraud_time])
        loc_dists = np.concatenate([normal_loc_dist, fraud_loc_dist])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        df = pd.DataFrame({
            'Transaction_Amount': amounts,
            'Time_Seconds': times,
            'Distance_From_Home': loc_dists,
            'Is_Fraud': labels
        })
        
        df['Transaction_Amount'] = df['Transaction_Amount'].clip(lower=1.0)
        df['Distance_From_Home'] = df['Distance_From_Home'].clip(lower=0.0)
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def train_models(self):
        try:
            self.df = self.generate_data()
            
            X = self.df.drop('Is_Fraud', axis=1)
            y = self.df['Is_Fraud']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            self.iso_forest = IsolationForest(contamination=0.02, random_state=42)
            self.iso_forest.fit(X_train_scaled)
            
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            self.clf.fit(X_train_scaled, y_train)
            
            self.status_var.set(f"Status: Models ready. Trained on {len(self.df)} synthetic transactions.")
            self.predict_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train models: {e}")
            self.status_var.set("Status: Error during training.")

    def predict(self):
        if not self.scaler:
            return
            
        try:
            amount = float(self.amount_var.get())
            time_hours = float(self.time_var.get())
            distance = float(self.distance_var.get())
            
            time_seconds = time_hours * 3600
            
            input_data = pd.DataFrame({
                'Transaction_Amount': [amount],
                'Time_Seconds': [time_seconds],
                'Distance_From_Home': [distance]
            })
            
            input_scaled = self.scaler.transform(input_data)
            
            # Supervised Prediction
            rf_pred = self.clf.predict(input_scaled)[0]
            rf_prob = self.clf.predict_proba(input_scaled)[0][1]
            
            if rf_pred == 1:
                self.rf_res_var.set(f"🚨 FRAUD DETECTED (Prob: {rf_prob:.1%})")
                self.rf_res_label.config(foreground="red")
            else:
                self.rf_res_var.set(f"✅ NORMAL (Fraud Prob: {rf_prob:.1%})")
                self.rf_res_label.config(foreground="green")
                
            # Unsupervised Prediction
            iso_pred = self.iso_forest.predict(input_scaled)[0]
            iso_mapped = 1 if iso_pred == -1 else 0
            
            if iso_mapped == 1:
                self.iso_res_var.set("🚨 ANOMALY DETECTED (Likely Fraud)")
                self.iso_res_label.config(foreground="red")
            else:
                self.iso_res_var.set("✅ NORMAL BEHAVIOR")
                self.iso_res_label.config(foreground="green")
                
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter valid numeric values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
