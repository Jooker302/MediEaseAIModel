import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np

# Load the dataset
data = pd.read_csv('deitplan_dataset.csv')

# Preprocessing
X = data.drop('Diet Plan', axis=1)
y = data['Diet Plan']
X = pd.get_dummies(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)

# Save the trained model
# sample_input_data = np.array([[30, 25.5, 120, 0, 1]])

# predicted_diet_plan = clf.predict(sample_input_data)

# print("Predicted Diet Plan:", predicted_diet_plan)

dump(clf, 'dietplan_model.pkl')
