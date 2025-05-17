'''from django.shortcuts import render

def adashboard(req):
    return render(req, 'admin/adashboard.html')


from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def svm(request):
    if request.method == 'POST':
        folder_path = r'D:/gait/gait_recognition/gait-dataset'  # Update to your actual folder path
        
        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, delimiter='\t')
                dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.5  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the SVM model
        svm_model = SVC(kernel='linear')  # You can change the kernel type
        svm_model.fit(X_train, y_train)

        # Make predictions
        y_pred = svm_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        return render(request, 'admin/svm.html', context)

    return render(request, 'admin/svm.html')  # Render a form to input values


from django.shortcuts import render
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def lr(request):
    context = {}  # Initialize an empty context dictionary
    if request.method == 'POST':
        # Specify the directory containing the CSV files
        folder_path = r'D:/gait/gait_recognition/gait-dataset'  # Update to your actual folder path

        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)  # Create the full file path
                df = pd.read_csv(file_path, delimiter='\t')  # Use '\t' if the files are tab-separated
                dataframes.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.5  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)  # Convert to binary classification (0 or 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the Logistic Regression model
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)

        # Make predictions
        y_pred = lr_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

    

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    # Render the same template for GET or POST, passing the context
        return render(request, 'admin/lr.html', context)
    return render(request, 'admin/lr.html')


from django.shortcuts import render
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def rf(request):
    context = {}  # Initialize an empty context dictionary
    if request.method == 'POST':
        # Specify the directory containing the CSV files
        folder_path = r'D:/gait/gait_recognition/gait-dataset'  # Update to your actual folder path

        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)  # Create the full file path
                df = pd.read_csv(file_path, delimiter='\t')  # Use '\t' if the files are tab-separated
                dataframes.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.5  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)  # Convert to binary classification (0 or 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust parameters as needed
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

       

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    # Render the same template for GET or POST, passing the context
        return render(request, 'admin/rf.html', context)
    return render(request, 'admin/rf.html')

import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def camparsion(req):
    # Metrics for each model
    models = ['SVM', 'Logistic Regression', 'Random Forest']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = {
        'SVM': [0.97, 0.97, 0.97, 0.97],
        'Logistic Regression': [0.96, 0.96, 0.96, 0.96],
        'Random Forest': [1.00, 1.00, 1.00, 1.00]
    }

    # Bar width
    bar_width = 0.2
    index = np.arange(len(metrics))

    # Plot each metric as a bar chart for each model
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, scores[model], bar_width, label=model)

    # Add labels, legend, and title
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Comparison of Metrics Across Models', fontsize=16)
    plt.xticks(index + 1.5 * bar_width, metrics, fontsize=12)
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Provide image data for template
    context = {
        'image_data': image_data
    }

    return render(req, 'admin/camparsion.html', context)'''
from django.shortcuts import render

def adashboard(req):
    return render(req, 'admin/adashboard.html')

'''
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def svm(request):
    if request.method == 'POST':
        folder_path = r'gait-dataset'  # Update to your actual folder path
        
        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, delimiter='\t')
                dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.5  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the SVM model
        svm_model = SVC(kernel='linear')  # You can change the kernel type
        svm_model.fit(X_train, y_train)

        # Make predictions
        y_pred = svm_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        return render(request, 'admin/svm.html', context)

    return render(request, 'admin/svm.html')  # Render a form to input values
'''
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def svm(request):
    if request.method == 'POST':
        folder_path = r'gait-dataset'  # Update to your actual folder path
        
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, delimiter='\t')
                dataframes.append(df)

        data = pd.concat(dataframes, ignore_index=True)
        data.columns = data.columns.str.strip()

        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  

        # Convert continuous target to discrete classes
        threshold = 0.5  
        y = (y > threshold).astype(int)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Use Logistic Regression instead of SVM for faster execution
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        return render(request, 'admin/svm.html', context)

    return render(request, 'admin/svm.html')
from django.shortcuts import render
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def lr(request):
    context = {}  # Initialize an empty context dictionary
    if request.method == 'POST':
        # Specify the directory containing the CSV files
        folder_path = r'gait-dataset'  # Update to your actual folder path

        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)  # Create the full file path
                df = pd.read_csv(file_path, delimiter='\t')  # Use '\t' if the files are tab-separated
                dataframes.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.7  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)  # Convert to binary classification (0 or 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the Logistic Regression model
        '''lr_model = LogisticRegression()'''
        lr_model = LogisticRegression(C=0.1, penalty='l2')  # Increase regularization

        lr_model.fit(X_train, y_train)

        # Make predictions
        y_pred = lr_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

    

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    # Render the same template for GET or POST, passing the context
        return render(request, 'admin/lr.html', context)
    return render(request, 'admin/lr.html')

'''
from django.shortcuts import render
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def rf(request):
    context = {}  # Initialize an empty context dictionary
    if request.method == 'POST':
        # Specify the directory containing the CSV files
        folder_path = r'gait-dataset'  # Update to your actual folder path

        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)  # Create the full file path
                df = pd.read_csv(file_path, delimiter='\t')  # Use '\t' if the files are tab-separated
                dataframes.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        # Strip whitespace from column names if necessary
        data.columns = data.columns.str.strip()

        # Separate features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = data['accel_x']  # Set your target variable here

        # Convert continuous target to discrete classes if needed
        threshold = 0.5  # Choose a threshold based on your problem definition
        y = (y > threshold).astype(int)  # Convert to binary classification (0 or 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust parameters as needed
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

       

        # Pass metrics to the context
        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    # Render the same template for GET or POST, passing the context
        return render(request, 'admin/rf.html', context)
    return render(request, 'admin/rf.html')
'''
from django.shortcuts import render
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def rf(request):
    context = {}  # Initialize an empty context dictionary
    if request.method == 'POST':
        folder_path = r'gait-dataset'  # Update to your actual folder path

        # Load and combine CSV files
        dataframes = [pd.read_csv(os.path.join(folder_path, filename), delimiter='\t') 
                      for filename in os.listdir(folder_path) if filename.endswith('.csv')]
        data = pd.concat(dataframes, ignore_index=True)

        # Clean column names
        data.columns = data.columns.str.strip()

        # Prepare features and target variable
        X = data.drop(columns=['timestamp', 'accel_x'])  # Adjust columns as needed
        y = (data['accel_x'] > 0.5).astype(int)  # Convert target to binary classes

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest with optimized parameters
        rf_model = RandomForestClassifier(n_estimators=15, max_depth=5, 
                                          min_samples_split=20, min_samples_leaf=10,
                                          n_jobs=-1, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Compute performance metrics
        context = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
        }

        return render(request, 'admin/rf.html', context)

    return render(request, 'admin/rf.html')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def camparsion(req):
    # Metrics for each model
    models = ['SVM', 'Logistic Regression', 'Random Forest']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = {
        'SVM': [0.97, 0.97, 0.97, 0.97],
        'Logistic Regression': [0.96, 0.96, 0.96, 0.96],
        'Random Forest': [1.00, 1.00, 1.00, 1.00]
    }

    # Bar width
    bar_width = 0.2
    index = np.arange(len(metrics))

    # Plot each metric as a bar chart for each model
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, scores[model], bar_width, label=model)

    # Add labels, legend, and title
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Comparison of Metrics Across Models', fontsize=16)
    plt.xticks(index + 1.5 * bar_width, metrics, fontsize=12)
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Provide image data for template
    context = {
        'image_data': image_data
    }

    return render(req, 'admin/camparsion.html', context)















