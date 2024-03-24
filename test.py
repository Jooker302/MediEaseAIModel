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
joblib.dump(clf, 'random_forest_model.pkl')


# Recommendations based on input data
def make_recommendations(input_data):
    # Predict whether the individual has diabetes
    prediction = clf.predict(input_data)
    return prediction


sample_data = [[45, 120, 25.5, 6.1,  # age, blood_glucose_level, bmi, HbA1c_level
                0, 0,  # hypertension, heart_disease
                0, 1, 0,  # gender (Female), smoking_history (No Info)
                0, 0, 0, 0, 0, 0]]  # One-hot encoded features

# Make predictions using the trained model
prediction, food, exercise, medicine = make_recommendations(sample_data)
print("Prediction (0 - No diabetes, 1 - Diabetes):", prediction)

