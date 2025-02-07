import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('diagnosis_trained_model.pkl', 'rb'))

# Creating function
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Make prediction without scaling
    prediction = loaded_model.predict(input_data_as_numpy_array)
    
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'


def main():
    st.title('Diabetes Prediction')
    
    # Collect user input
    Pregnancies = st.slider('Number of Pregnancies', 0, 20, 4, 1)
    Glucose = st.slider('Glucose level', 0, 199, 60, 1)
    BloodPressure = st.slider('Blood Pressure (mm Hg)', 0, 140, 80, 1)
    BMI = st.slider('BMI value', 0.0, 70.0, 33.3, 0.1)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.000, 3.0, 0.045, 0.001)
    Age = st.slider('Age', 10, 100, 21, 1)
    
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
