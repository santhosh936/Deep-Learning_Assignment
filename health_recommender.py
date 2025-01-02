import streamlit as st 
import pandas as pd 
# Load the dataset 
data = pd.read_csv('symptoms_data.csv') 
# Create a dictionary to map symptoms to diagnoses and treatments 
symptom_diagnosis_map = {} 
treatment_map = {} 
for index, row in data.iterrows(): 
    symptoms = row['symptoms'].split(', ') 
    diagnosis = row['diagnosis'] 
    treatment = row['treatment'] 
    for symptom in symptoms: 
        if symptom not in symptom_diagnosis_map: 
            symptom_diagnosis_map[symptom] = [] 
        symptom_diagnosis_map[symptom].append(diagnosis) 
        # Store the treatment for each diagnosis 
        treatment_map[diagnosis] = treatment 
def predict_diagnosis_and_treatment(user_input): 
    input_symptoms = [symptom.strip() for symptom in user_input.split(',')] 
    diagnosis_count = {} 
    for symptom in input_symptoms: 
        if symptom in symptom_diagnosis_map: 
            for diagnosis in symptom_diagnosis_map[symptom]: 
                if diagnosis not in diagnosis_count: 
                    diagnosis_count[diagnosis] = 0 
                diagnosis_count[diagnosis] += 1 
    if not diagnosis_count: 
        return "No diagnosis found for the given symptoms.", "" 
    # Find the diagnosis with the highest count 
    predicted_diagnosis = max(diagnosis_count, key=diagnosis_count.get) 
    predicted_treatment = treatment_map.get(predicted_diagnosis, "No treatment available")  
   return predicted_diagnosis, predicted_treatment 
# Streamlit app layout 
st.title('Simple Healthcare Symptoms Checker') 
# List all available symptoms 
available_symptoms = ', '.join(symptom_diagnosis_map.keys()) 
st.header("Available Symptoms") 
st.write(available_symptoms) 
# User input for symptoms 
user_input = st.text_input("Enter symptoms (comma separated):", "") 
if st.button('Check Diagnosis and Treatment'): 
    if user_input: 
        predicted_diagnosis, predicted_treatment = predict_diagnosis_and_treatment(user_input) 
        st.write(f"Possible Diagnosis: {predicted_diagnosis}") 
        st.write(f"Recommended Treatment: {predicted_treatment}") 
    else: 
        st.write("Please enter some symptoms.") 