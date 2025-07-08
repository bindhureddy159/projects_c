import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_excel('HealthCareData.xlsx')
df.columns = df.columns.str.strip()  # Strip extra spaces from column names

# Define the features to be used (must match model input)
features = [
    'Age', 'Gender', 'Place(location where the patient lives)', 'Duration of alcohol consumption(years)',
    'Quantity of alcohol consumption (quarters/day)', 'Type of alcohol consumed',
    'Hepatitis B infection', 'Hepatitis C infection', 'Diabetes Result', 'Blood pressure (mmhg)',
    'Obesity', 'Family history of cirrhosis/ hereditary', 'TCH', 'TG', 'LDL', 'HDL',
    'Hemoglobin  (g/dl)', 'PCV  (%)', 'RBC  (million cells/microliter)', 'MCV   (femtoliters/cell)',
    'MCH  (picograms/cell)', 'MCHC  (grams/deciliter)', 'Total Count', 'Polymorphs  (%)',
    'Lymphocytes  (%)', 'Monocytes   (%)', 'Eosinophils   (%)', 'Basophils  (%)',
    'Platelet Count  (lakhs/mm)', 'Total Bilirubin    (mg/dl)', 'Albumin   (g/dl)',
    'Direct    (mg/dl)', 'Indirect     (mg/dl)', 'Total Protein     (g/dl)', 'Globulin  (g/dl)', 'A/G Ratio',
    'AL.Phosphatase      (U/L)', 'SGOT/AST      (U/L)', 'SGPT/ALT (U/L)', 'USG Abdomen (diffuse liver or  not)'
]


# Keep only necessary columns
X = df[features].copy()

# Standardize common yes/no-type values (before encoding)
yes_no_cols = [
    'Hepatitis B infection', 'Hepatitis C infection', 'Diabetes Result', 'Obesity',
    'Family history of cirrhosis/ hereditary'
]

for col in yes_no_cols:
    X[col] = X[col].astype(str).str.strip().str.lower()
    X[col] = X[col].replace({'yes': 'positive', 'no': 'negative'})

# Label encode all categorical columns
# Save for decoding/transforming during prediction
label_encoders = {}
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].astype(str).str.strip().str.lower()
        le = LabelEncoder()

        if col == 'type of alcohol consumed':
            custom_classes = ['not applicable', 'country liquor', 'branded liquor', 'both']
            le.fit(custom_classes)
            print(f"✅ Forced alcohol classes: {le.classes_}")
            X[col] = X[col].apply(lambda x: x if x in custom_classes else 'not applicable')
        else:
            le.fit(X[col].unique())

        X[col] = le.transform(X[col])
        label_encoders[col] = le


# Normalize numeric values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted normalizer
with open('normalizer.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the fitted label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ normalizer.pkl and label_encoders.pkl saved successfully.")
