from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
import joblib  # Ensure joblib is installed globally

# Initialize Flask app
app = Flask(__name__)

# Load Excel data
chilli_diseases_df = pd.read_excel('chilli_diseases.xlsx')
diseases_pesticides_df = pd.read_excel('diseases_pesticides.xlsx')
health_effects_df = pd.read_excel('health_effects.xlsx')

# Load the trained model
rf_model = joblib.load('model.pkl')  # Load the saved Random Forest model
img_size = (150, 150)

# Define the categories based on your training data
categories = ['jalapeno', 'bhut_jolokia', 'kashmiri_chillie', 'cayenne_mirchi', 'guntur_sannam']

# Function to preprocess image
def preprocess_new_image(image_path, img_size):
    image = cv2.imread(image_path)
    if image is not None:
        image_resized = cv2.resize(image, img_size)
        image_normalized = image_resized / 255.0
        image_flattened = image_normalized.reshape(1, -1)
        return image, image_flattened
    else:
        return None, None

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the original file in the uploads folder
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Preprocess and classify image
    original_image, preprocessed_image = preprocess_new_image(file_path, img_size)
    if preprocessed_image is not None:
        rf_prediction = rf_model.predict(preprocessed_image)[0]
        predicted_variety = categories[rf_prediction]
        
        # Retrieve disease, pesticide, and health info
        info = get_info_by_variety(predicted_variety, chilli_diseases_df, diseases_pesticides_df, health_effects_df)
        
        # Save the original image in the static folder for display
        static_folder = 'static'
        os.makedirs(static_folder, exist_ok=True)
        static_image_path = os.path.join(static_folder, file.filename)
        cv2.imwrite(static_image_path, original_image)  # Save the original image

        return render_template('result.html', variety=predicted_variety, info=info, image=file.filename)
    else:
        return redirect(url_for('index'))

# Function to retrieve information
def get_info_by_variety(variety, chilli_diseases_df, diseases_pesticides_df, health_effects_df):
    diseases_row = chilli_diseases_df[chilli_diseases_df['Variety'] == variety]
    if diseases_row.empty:
        return f"No information available for {variety}"

    diseases = diseases_row.iloc[0]['Diseases'].split(',')
    info = [f"Variety: {variety}"]
    info.append(f"Diseases: {', '.join(diseases)}")

    for disease in diseases:
        pesticide_row = diseases_pesticides_df[diseases_pesticides_df['Disease'] == disease]
        if not pesticide_row.empty:
            pesticides_used = pesticide_row.iloc[0]['Pesticide used'].split(',')
            info.append(f"Disease: {disease}")
            info.append(f"Pesticide(s) Used: {', '.join(pesticides_used)}")
            
            for pesticide in pesticides_used:
                # Clean pesticide name
                pesticide = pesticide.strip()  # Remove leading/trailing spaces
                health_row = health_effects_df[health_effects_df['Pesticide'] == pesticide]

                if not health_row.empty:
                    health_effects = health_row.iloc[0]['Health Effects']  # Ensure this column exists
                    info.append(f"Health Effects of {pesticide}: {health_effects}")
                else:
                    info.append(f"Health Effects of {pesticide}: No data available")

    return "<br>".join(info) # Return as a single string



if __name__ == "__main__":
    app.run(debug=True)
