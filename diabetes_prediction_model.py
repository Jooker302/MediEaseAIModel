import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib

# Read the dataset
diabetes = pd.read_csv('diabetes_prediction_dataset.csv')

# One-hot encode categorical variables
diabetes_encoded = pd.get_dummies(diabetes, columns=['gender', 'smoking_history'])

# Prepare features (X) and target variable (y)
X = diabetes_encoded.drop(columns=['diabetes'])
y = diabetes_encoded['diabetes']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print("Accuracy of the model on the test set:", accuracy)


# Save the trained model to a file
joblib.dump(clf, 'diabetes_prediction_model.pkl')

