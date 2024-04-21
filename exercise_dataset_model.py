import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv("exercise_dataset.csv")

# Preprocessing
# One-hot encode the 'Gender' column
data = pd.get_dummies(data, columns=['Gender'])

# Feature selection
X = data[['Age', 'Gender_Male', 'Gender_Female', 'BMI']]
y = data['Exercise']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# feature_names = X_train.columns.tolist()

# print("Input features required by the model:", feature_names)
# Evaluate model
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)

# Single sample data
# sample = {
#     'Age': 30,            # Sample age
#     'Gender_Male': 1,     # Sample gender (Male = 1)
#     'Gender_Female': 0,   # Sample gender (Female = 0)
#     'BMI': 27.8           # Sample BMI value
# }

# Create a DataFrame from the single sample
# sample_df = pd.DataFrame([sample])

# Make prediction for the single sample
# prediction = model.predict(sample_df)
joblib.dump(model, 'exercise_model.pkl')

# Display the prediction
# print("Prediction for the single sample:", prediction[0])