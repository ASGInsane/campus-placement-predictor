import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("placement_data.csv")

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
college_encoder = LabelEncoder()
df['college'] = college_encoder.fit_transform(df['college'])

# Map target variable
df['placed'] = df['placed'].map({'Placed': 1, 'Not Placed': 0})

# Features and target
X = df[['cgpa', 'gender', 'tenth', 'twelfth', 'backlogs', 'degree', 'college', 'interview_score']]
y = df['placed']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save college encoder too (optional for reverse-mapping later)
with open("college_encoder.pkl", "wb") as f:
    pickle.dump(college_encoder, f)

