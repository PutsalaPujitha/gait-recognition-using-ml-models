# Machine Learning-Based Exploration and Biometric Analysis for Gait Recognition

## Overview

This project is a full-stack web application designed to analyze and recognize human gait patterns using biometric data and machine learning models. It assists researchers and healthcare professionals in identifying individuals or detecting gait abnormalities based on gait parameters.

The system enables users to input gait-related data and obtain predictive results using trained machine learning models. Additionally, the application provides an admin dashboard for model management and result tracking.

## Features

- User registration and login
- Input form for gait-related biometric data (e.g., cadence, walking speed, stride length)
- Gait prediction using ML models: SVM, Random Forest, Logistic Regression
- Result visualization and status (recognized/not recognized)
- Admin panel to manage users, view reports, and compare model outputs
- Secure, lightweight web interface with a responsive UI

## Tech Stack

**Frontend:**
- HTML5, CSS3
- JavaScript, Bootstrap

**Backend:**
- Python (Django Framework)

**Machine Learning Models:**
- Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression

**Database:**
- SQLite (for local testing and deployment)

## Machine Learning Models

The following models are used for classification and recognition:

- **SVM (Support Vector Machine):**
  Effective in high-dimensional spaces, particularly suitable for gait data classification.

- **Random Forest:**
  Ensemble learning technique that improves accuracy and reduces overfitting.

- **Logistic Regression:**
  Lightweight and interpretable baseline model for binary gait classification.

All models were trained on a gait dataset consisting of features such as cadence, stride length, step width, walking speed, and other biometric parameters.

