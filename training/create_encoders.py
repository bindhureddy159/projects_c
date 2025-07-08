import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your cleaned dataset
df = pd.read_excel("HealthCareData.xlsx")

# Specify categorical columns to encode
categorical_cols = [
    'Gender', 'Place(location where the patient lives)', 'Type of alcohol consumed',
    'Hepatitis B infection', 'Hepatitis C infection', 'Diabetes Result', 'Obesity',
    'Family history of cirrhosis/ hereditary', 'Blood pressure (mmhg)',
    'USG Abdomen (diffuse liver or  not)'
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.lower().str.strip()
    if col == 'Type of alcohol consumed':
        custom_classes = ['country liquor', 'branded liquor', 'both', 'not applicable']
        le.fit(custom_classes)
        print(f"✅ Forced classes for '{col}':", le.classes_)
        df[col] = df[col].apply(lambda x: x if x in custom_classes else 'not applicable')
        df[col] = le.transform(df[col])

    else:
        le.fit(df[col].unique())
        df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save the encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Label encoders created and saved successfully.")
