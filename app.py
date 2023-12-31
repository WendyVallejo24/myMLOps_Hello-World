import streamlit as st
import requests
import json
import numpy as np

def predict_linear_model(input_values):
    server_url = 'https://linear-model-service-wendyvallejo24.cloud.okteto.net/v1/models/linear-model:predict'

    payload = {'instances': [[value] for value in input_values]}
    print("Payload:", payload)
    response = requests.post(server_url, data=json.dumps(payload))
    response.raise_for_status()
    prediction = response.json()
    return prediction

def main():
    st.title('Linear Model Prediction')

    # Input values
    valor1 = st.number_input("Enter value 1 to predict: ", value=0.0)
    valor2 = st.number_input("Enter value 2 to predict: ", value=0.0)
    valor3 = st.number_input("Enter value 3 to predict: ", value=0.0)

    if st.button("Predict"):
        # Make prediction
        valores = valor1, valor2, valor2
        prediction = predict_linear_model(valores)

        # Display prediction
        st.success(f"{prediction}")


if __name__ == '__main__':
    main()