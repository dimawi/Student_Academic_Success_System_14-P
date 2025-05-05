# Student_Academic_Success_System_14-P

This repository contains a machine learning project designed to predict student performance using various machine learning models. The dataset comprises student performance data with features such as study hours, family background, and test scores. The project employs different models, such as Logistic Regression, Random Forest, and Decision Trees, to predict students' scores and categorize them into different performance levels.

## Project Overview
The goal of this project is to predict the performance of students based on multiple factors. The model is trained using student data to forecast performance outcomes, which can be used to provide better insights into student improvement areas.

Key Features:
Prediction of student performance using different machine learning models.

Evaluation of model performance using accuracy, precision, recall, and F1 score.

Data preprocessing, feature scaling, and splitting techniques.

Use of a Streamlit app to provide a user-friendly interface for making predictions.

## Requirements
To run this project, you will need Python 3.7 or higher. You can install the required dependencies by using the following command:
```bash
pip install -r requirements.txt
```
The following libraries are required:

Pandas

NumPy

scikit-learn

Streamlit

joblib

Matplotlib

Seaborn

## Project Structure
Here is the structure of the project:
```bash
.
├── .venv/                  # Virtual environment directory (created using venv)
│    ├── data/              # Folder containing raw data files
│    │    ├── Reports (23).xlsx
│    │    ├── student_data.db
│    │    └── The_Final_Dataset.xlsx
│    ├── src/               # Source code and Jupyter notebooks
│    │    ├── Final-3.ipynb  # Jupyter Notebook for model development
│    │    ├── FinalPM.ipynb  # Notebook for performance metrics analysis
│    │    ├── streamlitApp.py  # Streamlit app for the model's user interface
│    │    └── streamlitAppSQLite.py  # Streamlit app using SQLite for storing results
│    └── Trained models/     # Folder containing saved models and related files
│         ├── Final-3.ipynb  # Final model training and evaluation
│         ├── columns.joblib  # Column transformer for feature preprocessing
│         ├── decision_tree_model.joblib  # Saved Decision Tree model
│         ├── logistic_regression_model.joblib  # Saved Logistic Regression model
│         ├── random_forest_model.joblib  # Saved Random Forest model
│         └── scaler.joblib  # Scaler used for feature scaling
├── README.md               # This README file
├── requirements.txt        # Python package dependencies file
├── .gitignore              # Git ignore file
└── pyvenv.cfg              # Virtual environment configuration
```
## Data Description
The dataset contains data related to student performance. The files inside the data/ folder are as follows:

Reports (23).xlsx: An Excel file containing detailed reports of student performance.

student_data.db: A SQLite database containing student-related data.

The_Final_Dataset.xlsx: The final cleaned and preprocessed dataset used for model training.

## Model Architecture
This project utilizes several machine learning models to predict student performance:

Decision Tree Classifier: A model that classifies students into different performance categories based on various features.

Logistic Regression: A classification algorithm used to predict binary outcomes.

Random Forest Classifier: An ensemble model for classification tasks, offering better performance than individual decision trees.

The models are trained and evaluated on the dataset, and the best-performing model is selected for predictions.

## Streamlit Application
A Streamlit application is included to provide a user interface for the prediction system. The app allows users to input student data, such as study hours and test scores, and get a prediction about their performance.

streamlitApp.py: The main Streamlit app file that provides the front-end interface for users.

streamlitAppSQLite.py: A variant of the Streamlit app that uses SQLite to store user predictions.

## Training & Model Evaluation
Hyperparameters:
The models are trained with the following parameters:

Decision Tree: Default hyperparameters used, with tuning options for depth and splits.

Logistic Regression: Regularization parameter (C) is tuned for optimal performance.

Random Forest: Tuned for the number of trees (n_estimators) and maximum depth.

Model Evaluation Metrics:
The model performance is evaluated based on the following metrics:

Accuracy: Percentage of correct predictions.

Precision: Measure of how many predicted positives are actual positives.

Recall: Measure of how many actual positives are captured by the model.

F1 Score: Harmonic mean of precision and recall.

These metrics are used to compare model performance and determine which one is best suited for predicting student outcomes.

## Usage
1. Run the Streamlit App:
To run the Streamlit app, use the following command:
```bash
streamlit run src/streamlitApp.py
```
This will launch a web interface where you can input student data and get a prediction.

3. Train and Evaluate Models:
To train and evaluate the models, run the following Jupyter notebooks:

Final-3.ipynb: The notebook for model training and evaluation.

FinalPM.ipynb: The notebook for analyzing model performance.

3. Saving and Loading Models:
Once the models are trained, they are saved as .joblib files in the Trained models/ folder. These models can be loaded and used for making predictions.

For example, to load a model and make predictions:
```bash
from joblib import load

# Load the trained model
model = load('Trained models/random_forest_model.joblib')

# Make a prediction
prediction = model.predict([[study_hours, test_score]])
```
## Results
Once the models are trained and evaluated, you will get the following results:

Training and evaluation metrics such as accuracy, precision, recall, and F1 score.

The best-performing model will be used for making future predictions.

Visualizations for model performance (e.g., confusion matrix, classification reports).

## Acknowledgements
The dataset used in this project is publicly available.

The project leverages machine learning algorithms from scikit-learn.

Special thanks to Streamlit for providing an easy-to-use framework for the web interface.
