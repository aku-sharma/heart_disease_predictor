import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
data = pd.read_csv('data/heart.csv')

# Features & target (use exact CSV column names)
X = data[['Age','Sex','Cp','Trestbps','Chol','Fbs','Restecg','Thalach','Exang','Oldpeak','Slope','Ca','Thal']]
y = data['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & scaler
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)