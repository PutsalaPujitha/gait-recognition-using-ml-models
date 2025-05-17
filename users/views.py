
from django.shortcuts import render

def udashboard(req):
    return render(req, 'user/udashboard.html')


# Load the model and scaler
import numpy as np
import pickle
from django.shortcuts import render

# Combined function to load the model, process input, and make predictions
def prediction(request):
    result = None
    if request.method == 'POST':
        # Load the model and scaler inside the predict function
        with open('rf_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        # Collect features from the request
        features = [
            float(request.POST['left_cadence']),
            float(request.POST['right_cadence']),
            float(request.POST['walking_speed']),
            float(request.POST['stride_length']),
            float(request.POST['step_width']),
            float(request.POST['limp_index']),
            float(request.POST['additional_feature_1']),
            float(request.POST['additional_feature_2']),
            float(request.POST['additional_feature_3']),
            float(request.POST['additional_feature_4']),

          
           
        ]

        # Preprocess user input for prediction
        user_data = np.array(features).reshape(1, -1)
        
        if user_data.shape[1] != 10:
            raise ValueError("Expected 10 features but got a different number.")
        
        user_data = scaler.transform(user_data)  # Transform the input data
        prediction = model.predict(user_data)[0]  # Make prediction
        
        # Interpret the result
        result = "Gait recognized" if prediction == 1 else "Gait not recognized"

    return render(request, 'user/prediction.html', {'result': result})

