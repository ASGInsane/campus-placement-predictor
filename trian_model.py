import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('placement_data.csv')

# Encode categorical columns
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['specialisation'] = le.fit_transform(df['specialisation'])
df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})

# Split features and target
X = df.drop('status', axis=1)
y = df['status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
