import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_excel("HealthCareData.xlsx")
df.columns = df.columns.str.strip()

# Define target
target = 'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)'
df[target] = df[target].astype(str).str.strip().str.lower()
df[target] = df[target].replace({'yes': 1, 'no': 0})
print("📊 Unique values in target after cleaning:", df[target].unique())

# Filter out only rows with target 0 or 1
df = df[df[target].isin([0, 1])]

print("📊 Rows with valid target (0/1):", len(df))

# Numeric columns that may contain dirty data
numeric_columns = [
    'TCH', 'TG', 'LDL', 'HDL',
    'Hemoglobin  (g/dl)', 'PCV  (%)', 'RBC  (million cells/microliter)',
    'MCV   (femtoliters/cell)', 'MCH  (picograms/cell)', 'MCHC  (grams/deciliter)',
    'Total Count', 'Polymorphs  (%)', 'Lymphocytes  (%)', 'Monocytes   (%)',
    'Eosinophils   (%)', 'Basophils  (%)', 'Platelet Count  (lakhs/mm)',
    'Total Bilirubin    (mg/dl)', 'Direct    (mg/dl)', 'Indirect     (mg/dl)',
    'Total Protein     (g/dl)', 'Albumin   (g/dl)', 'Globulin  (g/dl)',
    'A/G Ratio', 'AL.Phosphatase      (U/L)', 'SGOT/AST      (U/L)', 'SGPT/ALT (U/L)'
]

# Convert dirty numerics to float, coercing errors to NaN
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows with too many NaNs in these columns
df = df[df[numeric_columns].isnull().sum(axis=1) <= 5]
print("📉 Rows after dropping rows with >5 NaNs in numeric fields:", df.shape[0])

# Fill remaining NaNs with column means
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Define features (drop S.NO and target)
features = [col for col in df.columns if col not in ['S.NO', target]]
X = df[features].copy()
y = df[target].astype(int)

# Label encode categorical columns
label_encoders = {}

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str).str.strip().str.lower()
        le = LabelEncoder()

        if col == 'Type of alcohol consumed':
            custom_classes = ['country liquor', 'branded liquor', 'both', 'not applicable']
            le.fit(custom_classes)
            print(f"✅ Forced alcohol classes: {le.classes_}")
            X[col] = X[col].apply(lambda x: x if x in custom_classes else 'not applicable')
        else:
            le.fit(X[col].unique())

        X[col] = le.transform(X[col])
        label_encoders[col] = le

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42,stratify=y)




# Train the model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Save model and tools
pickle.dump(model, open("liver_prediction.pkl", "wb"))
pickle.dump(scaler, open("normalizer.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
print("✅ Model, normalizer, and encoders saved.")




