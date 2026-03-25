import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# Cached Data Generation
@st.cache_data
def load_and_preprocess_data(n_samples=15000, fraud_ratio=0.02):
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
    
    df['Transaction_Amount'] = df['Transaction_Amount'].clip(lower=1.0)
    df['Distance_From_Home'] = df['Distance_From_Home'].clip(lower=0.0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

@st.cache_resource
def train_models(df):
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    return scaler, iso_forest, clf

# UI Structure
st.title("🛡️ Banking Transaction Fraud Detection")
st.markdown("This dashboard uses **Machine Learning** (Random Forest) and **Anomaly Detection** (Isolation Forest) to detect fraudulent transactions.")

with st.spinner('Loading data and training models...'):
    df = load_and_preprocess_data()
    scaler, iso_forest, clf = train_models(df)

# Sidebar for Input
st.sidebar.header("Input Transaction Details")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
time_hours = st.sidebar.slider("Time of Transaction (Hour of Day)", 0.0, 23.99, 12.0)
distance = st.sidebar.number_input("Distance From Home (Miles)", min_value=0.0, max_value=5000.0, value=25.0, step=1.0)

time_seconds = time_hours * 3600

if st.sidebar.button("Predict Fraud", use_container_width=True):
    # Predict
    input_data = pd.DataFrame({
        'Transaction_Amount': [amount],
        'Time_Seconds': [time_seconds],
        'Distance_From_Home': [distance]
    })
    
    input_scaled = scaler.transform(input_data)
    
    # Random Forest Prediction
    rf_pred = clf.predict(input_scaled)[0]
    rf_prob = clf.predict_proba(input_scaled)[0][1]
    
    # Isolation Forest Prediction
    iso_pred = iso_forest.predict(input_scaled)[0]
    iso_mapped = 1 if iso_pred == -1 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest (Supervised)")
        if rf_pred == 1:
            st.error(f"🚨 FRAUD DETECTED\n\nProbability: {rf_prob:.1%}")
        else:
            st.success(f"✅ NORMAL TRANSACTION\n\nFraud Probability: {rf_prob:.1%}")
            
    with col2:
        st.subheader("Isolation Forest (Anomaly Detection)")
        if iso_mapped == 1:
            st.error("🚨 ANOMALY DETECTED (Likely Fraud)")
        else:
            st.success("✅ NORMAL BEHAVIOR")

st.markdown("---")
st.subheader("Dataset Overview")
st.write(f"The model was trained on a synthetic dataset of **{len(df):,}** transactions (Fraud Ratio: 2%).")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    fig1 = px.scatter(df.head(1000), x='Transaction_Amount', y='Distance_From_Home', color='Is_Fraud', 
                      title='Transaction Amount vs Distance (Sample of 1000)',
                      color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig1, use_container_width=True)

with col_viz2:
    fig2 = px.histogram(df, x='Transaction_Amount', color='Is_Fraud', barmode='overlay', 
                        title='Distribution of Transaction Amounts',
                        color_discrete_map={0: 'blue', 1: 'red'})
    fig2.update_xaxes(range=[0, 3000])
    st.plotly_chart(fig2, use_container_width=True)
