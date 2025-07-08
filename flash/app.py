from flask import Flask, request, render_template
import pickle
import numpy as np
# import pandas as pd
app = Flask(__name__)
# Load model, normalizer, and encoders
model = pickle.load(open('liver_prediction.pkl', 'rb'))
normalizer = pickle.load(open('normalizer.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
alcohol_encoder = label_encoders['Type of alcohol consumed']
print("🔍 Classes for 'Type of alcohol consumed':", alcohol_encoder.classes_)
print("✅ Model loaded:", model)
# Exact 41 field names as per your dataset
fields = [
    'Age','Gender','Place(location where the patient lives)','Duration of alcohol consumption(years)','Quantity of alcohol consumption (quarters/day)','Type of alcohol consumed',
    'Hepatitis B infection','Hepatitis C infection','Diabetes Result','Blood pressure (mmhg)','Obesity','Family history of cirrhosis/ hereditary',
    'TCH','TG','LDL','HDL','Hemoglobin  (g/dl)','PCV  (%)','RBC  (million cells/microliter)','MCV   (femtoliters/cell)','MCH  (picograms/cell)',
    'MCHC  (grams/deciliter)','Total Count','Polymorphs  (%)','Lymphocytes  (%)','Monocytes   (%)','Eosinophils   (%)','Basophils  (%)',
    'Platelet Count  (lakhs/mm)','Total Bilirubin    (mg/dl)','Direct    (mg/dl)','Indirect     (mg/dl)','Total Protein     (g/dl)','Albumin   (g/dl)',
    'Globulin  (g/dl)','A/G Ratio','AL.Phosphatase      (U/L)','SGOT/AST      (U/L)','SGPT/ALT (U/L)','USG Abdomen (diffuse liver or  not)'
]
@app.route('/')
def home():
    form_fields = []
    for field in fields:
        if field in label_encoders:
            options = label_encoders[field].classes_
            select_html = f'<select name="{field}">' + ''.join([
                f'<option value="{opt}">{opt}</option>' for opt in options
            ]) + '</select>'
            form_fields.append({'label': field, 'input': select_html})
        else:
            input_html = f'<input type="text" name="{field}"'
            if field == 'Blood pressure (mmhg)':
                input_html += ' placeholder="e.g. 120/80" pattern="\\d{2,3}/\\d{2,3}" title="Enter like 120/80"'
            input_html += '>'
            form_fields.append({'label': field, 'input': input_html})
    return render_template('index.html', form_fields=form_fields)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    try:
        for field in fields:
            value = request.form.get(field, '').strip()

            if field == 'Blood pressure (mmhg)':
                # Expect value like "120/80" and convert to 120.80 float
                if '/' not in value:
                    return "❌ Please enter blood pressure in the format 120/80"
                try:
                    systolic, diastolic = map(int, value.split('/'))
                    combined_bp = float(f"{systolic}.{diastolic}")  # e.g. 120.80
                    input_data.append(combined_bp)
                except:
                    return "❌ Invalid blood pressure value. Use format like 120/80"
                continue

           
            if field in label_encoders:
                encoder = label_encoders[field]
                encoder_classes = [cls.lower().strip() for cls in encoder.classes_]

                if value.lower() not in encoder_classes:
                    return f"❌ Invalid categorical input for '{field}': '{value}'. Expected: {list(encoder.classes_)}"

                matched_value = encoder.classes_[encoder_classes.index(value.lower())]
                encoded = encoder.transform([matched_value])[0]
                input_data.append(encoded)
            else:
                try:
                    input_data.append(float(value))
                except ValueError:
                    return f"❌ Invalid numeric input for '{field}': '{value}'"
        if len(input_data) != len(fields):
            return f"❌ Expected {len(fields)} inputs, got {len(input_data)}."

        input_array = np.array([input_data])
        normalized = normalizer.transform(input_array)
        prediction = model.predict(normalized)[0]
        print("🧪 Encoded input:", input_data)
        print("🧪 Prediction output:", prediction)

        result_text = "🟢 Result: Not Likely Cirrhosis"
        result_class = ""
        if str(prediction).strip().lower() in ['1', 'yes']:
            result_text = "🔴 Result: Likely Cirrhosis"
            result_class = "danger"

        return render_template('index.html', form_fields=[
            {'label': field, 'input': f'<input type="text" name="{field}" value="{request.form.get(field,"")}>' if field not in label_encoders else ''}
            for field in fields
        ], prediction_text=result_text, result_class=result_class)

    except Exception as e:
        return f"⚠️ Error: {str(e)}"
if __name__ == '__main__':  
    app.run(debug=True) 