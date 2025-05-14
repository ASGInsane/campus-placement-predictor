import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('placements.csv')

# Encode gender: Male=1, Female=0
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Encode college (Label Encoding)
college_mapping = {name: idx for idx, name in enumerate(df['college'].unique())}
df['college'] = df['college'].map(college_mapping)

# Features and target
X = df[['cgpa', 'gender', 'tenth', 'twelfth', 'backlogs', 'degree', 'college', 'interview_score']]
y = df['placed']

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
