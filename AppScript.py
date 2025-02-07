# -*- coding: utf-8 -*-


import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('diagnosis_trained_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure you saved and load the scaler used during training

# Creating function
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Scale input data using the same scaler from training
    input_data_scaled = scaler.transform(input_data_as_numpy_array)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_scaled)
    
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
    

    
    
def main():
    
    st.title('Diabetes Prediction')
    
  
    
    Pregnancies=st.slider('Number of Pregnancies',min_value=0,max_value=20,value=4,step=1)
    Glucose=st.slider('Glucose level',min_value=0,max_value=199,value=60,step=1)
    BloodPressure=st.slider('BloodPressure level in mm Hg',min_value=0,max_value=140,value=80,step=1)
    BMI=st.slider('BMI value',min_value=0.0,max_value=70.0,value=33.3,step=0.01)
    DiabetesPedigreeFunction=st.slider('Diabetes Pedigree Function value',min_value=0.000, step=0.001, max_value=3.0, value=0.045, format="%3f")
    Age=st.slider('Age of a person',min_value=10,max_value=100,value=21,step=1)
   
    
   
    diagnosis = ''
    if st.button('Dibetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)    
    


if __name__ == '__main__':
     main()    
    
    
