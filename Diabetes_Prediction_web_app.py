import numpy as np
import pickle
import pandas as pd
import streamlit as st

# loading the saved model
loaded_model = pickle.load( open( 'trained_model.sav','rb'))

#pickle_in = open("trained_model.pkl","rb")
#model = pickle.load(pickle_in)



# creating a function forn Prediction

def diabetes_prediction(input_data):
     
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  
    prediction = loaded_model.predict(input_data_reshaped)
  

    if(prediction[0]==0):
      return 'The persion is Non Diabetic'
    else:
      return 'The persion is Diabetic'
  


def main():

    # giving a title 
    st.title('Diabetes Prediction Web App')  

    
    # getting the data from user
    Pregnancies = st.text_input('Number of Pregenancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure =st.text_input('BloodPressure value')
    SkinThickness =st.text_input('SkinThickness value')
    Insulin =st.text_input('Insulin level')
    BMI =st.text_input('BMI value')
    DiabetesPedigreeFunction =st.text_input('DiabetesPedigreeFunction value')
    Age =st.text_input('Age of the Person')
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction((Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age))
   
    st.success(diagnosis)   
    

if __name__ == '__main__':
    main()    
