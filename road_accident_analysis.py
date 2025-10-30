#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("✅ All libraries imported successfully!")


# In[2]:


import pandas as pd
import numpy as np

# Create a synthetic dataset for road accident severity prediction
data = {
    'Age_band_of_driver': np.random.choice(['Under 18', '18-30', '31-50', 'Over 50'], 200),
    'Sex_of_driver': np.random.choice(['Male', 'Female'], 200),
    'Educational_level': np.random.choice(['None', 'High School', 'College', 'University'], 200),
    'Vehicle_driver_relation': np.random.choice(['Owner', 'Employee', 'Other'], 200),
    'Driving_experience': np.random.choice(['Below 5yr', '5-10yr', 'Above 10yr'], 200),
    'Type_of_vehicle': np.random.choice(['Car', 'Motorcycle', 'Bus', 'Truck'], 200),
    'Area_accident_occured': np.random.choice(['Urban', 'Rural'], 200),
    'Day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 200),
    'Time_to_accident': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 200),
    'Lanes_or_Medians': np.random.choice(['One lane', 'Two lanes', 'More than two'], 200),
    'Road_surface_type': np.random.choice(['Asphalt', 'Gravel', 'Dirt'], 200),
    'Light_conditions': np.random.choice(['Daylight', 'Darkness'], 200),
    'Weather_conditions': np.random.choice(['Clear', 'Rainy', 'Foggy'], 200),
    'Number_of_casualties': np.random.randint(1, 5, 200),
    'Accident_severity': np.random.choice(['Slight Injury', 'Serious Injury', 'Fatal'], 200)
}

df = pd.DataFrame(data)
df.to_csv('accidents.csv', index=False)

print("✅ Synthetic dataset created and saved as 'accidents.csv'")
df.head()


# In[3]:


# Check basic info
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n")
print(df.dtypes)
print("\nMissing values:\n")
print(df.isnull().sum())

# View some unique values
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}: {df[col].unique()[:5]}")


# In[7]:


# Step 6: Data Preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣  Encode categorical columns
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le  # Save encoders for later decoding if needed

print("✅ All categorical columns encoded.\n")

# 2️⃣  Split features (X) and target (y)
X = df_encoded.drop('Accident_severity', axis=1)
y = df_encoded['Accident_severity']

print(f"Feature set shape: {X.shape}")
print(f"Target shape: {y.shape}\n")

# 3️⃣  Split data into training & testing sets (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data split complete:")
print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}\n")

#


# In[9]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling complete.")


# In[10]:


# Step 7: Train and Evaluate Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Initialize and train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Model training complete.\n")

# 2️⃣ Make predictions
y_pred = model.predict(X_test_scaled)

# 3️⃣ Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Model Accuracy: {accuracy * 100:.2f}%\n")

# 4️⃣ Detailed classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 5️⃣ Confusion Matrix visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[11]:


# Step 8: Make Real-Time Predictions

import numpy as np
import pandas as pd

# ✅ 1️⃣ Create a new accident record (example input)
new_accident = {
    'Age_band_of_driver': '31-50',
    'Sex_of_driver': 'Male',
    'Educational_level': 'College',
    'Vehicle_driver_relation': 'Owner',
    'Driving_experience': '5-10yr',
    'Type_of_vehicle': 'Car',
    'Area_accident_occured': 'Urban',
    'Day_of_week': 'Friday',
    'Time_to_accident': 'Evening',
    'Lanes_or_Medians': 'Two lanes',
    'Road_surface_type': 'Asphalt',
    'Light_conditions': 'Daylight',
    'Weather_conditions': 'Clear',
    'Number_of_casualties': 2
}

# ✅ 2️⃣ Convert to DataFrame
new_df = pd.DataFrame([new_accident])

# ✅ 3️⃣ Encode using the same label encoders used during training
for col, le in label_encoders.items():
    if col in new_df.columns:
        new_df[col] = le.transform(new_df[col])

# ✅ 4️⃣ Scale the new data using the same scaler
new_scaled = scaler.transform(new_df)

# ✅ 5️⃣ Make prediction
prediction = model.predict(new_scaled)[0]

# ✅ 6️⃣ Decode numeric prediction back to label
predicted_label = label_encoders['Accident_severity'].inverse_transform([prediction])[0]

print("🚦 Predicted Accident Severity:", predicted_label)


# In[ ]:




