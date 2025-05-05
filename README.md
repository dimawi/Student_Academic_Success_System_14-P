# Student Performance Prediction System

This repository contains a machine learning project designed to predict student performance based on various factors such as gender, age, GPA, and other demographic variables. Using machine learning models like Logistic Regression, Decision Trees, and Random Forest, this system predicts whether a student is likely to succeed or fail academically.

## Project Overview

The project is focused on predicting student performance using multiple machine learning algorithms and a clean dataset with features such as:

- **Gender**: Male or Female
- **Age**: Age of the student
- **Attestat GPA**: Grade Point Average
- **Status**: Active or inactive student
- **School Type**: Type of school the student attends (School or Lycee)
- **Nationality**: One-hot encoded nationality features (e.g., Kazakh)

The project leverages different models and evaluates their accuracy in terms of predicting student performance. The results include classifications of "likely to succeed" or "likely to fail" based on the model's predictions.

## Requirements

To run this project, you will need Python 3.7 or higher. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```
The following libraries are required:

- **Pandas**

- **NumPy**

- **scikit-learn**

- **Streamlit**

- **joblib**

- **Matplotlib**

- **Seaborn**

## Technology Stack
### 1. Programming Language
**Python** – Core language used for data manipulation, analysis, modeling, and application deployment.

### 2. Libraries & Frameworks
**Pandas** – Data manipulation and analysis (tabular data).

**NumPy** – Numerical computing and array operations.

**Scikit-learn** – Machine learning models and evaluation metrics.

**Matplotli**b – Data visualization using static plots.

**Seaborn** – Statistical data visualization built on top of Matplotlib.

**Streamlit** – Web application framework for deploying interactive ML apps.

**joblib** – Model serialization and persistence (saving/loading models).

### 3. Development Tools & Environments
**PyCharm** – Integrated Development Environment (IDE) for Python programming.

**Jupyter Notebook** – Interactive computing environment, often used for prototyping, data exploration, and documentation.

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
The dataset contains data related to student performance. The files inside the ```data/``` folder are as follows:

**Reports (23).xlsx:** An Excel file containing detailed reports of student performance.

**student_data.db:** A SQLite database containing student-related data.

**The_Final_Dataset.xlsx:** The final cleaned and preprocessed dataset used for model training.

## Data Processing
The data preprocessing pipeline includes:

**One-Hot Encoding:** Categorical columns such as nationality are one-hot encoded.

**Feature Scaling:** Data scaling is applied to numeric features (like GPA) to standardize them before training models.

**Data Splitting:** The dataset is split into training, validation, and test sets to prevent overfitting and evaluate performance.

## Model Architecture
This project utilizes several machine learning models to predict student performance:

**Decision Tree Classifier:** A model that classifies students into different performance categories based on various features.

**Logistic Regression:** A classification algorithm used to predict binary outcomes.

**Random Forest Classifier:** An ensemble model for classification tasks, offering better performance than individual decision trees.

The models are trained and evaluated on the dataset, and the best-performing model is selected for predictions.

## Model Evaluation
Here are the performance results of the models:

**Decision Tree Accuracy:** 88.45%

**Logistic Regression Accuracy:** 88.88%

**Random Forest Accuracy:** 89.57%

The evaluation also includes detailed classification reports for each model:
Decision Tree Classification Report:
```bash
              precision    recall  f1-score   support
           0       0.82      0.86      0.84       405
           1       0.92      0.90      0.91       755

    accuracy                           0.88      1160
   macro avg       0.87      0.88      0.87      1160
weighted avg       0.89      0.88      0.89      1160
```
Logistic Regression Classification Report:
```bash
              precision    recall  f1-score   support
           0       0.87      0.80      0.83       405
           1       0.90      0.93      0.92       755

    accuracy                           0.89      1160
   macro avg       0.88      0.87      0.88      1160
weighted avg       0.89      0.89      0.89      1160
```
Random Forest Classification Report:
```bash
              precision    recall  f1-score   support
           0       0.84      0.87      0.85       405
           1       0.93      0.91      0.92       755

    accuracy                           0.90      1160
   macro avg       0.88      0.89      0.89      1160
weighted avg       0.90      0.90      0.90      1160
```

## Streamlit Application
The project includes a **Streamlit application** that allows users to input student data (such as gender, age, GPA, etc.) and get predictions on whether the student is likely to succeed or fail academically.

Before running streamlit you have opportunity to run the application using this link:
```link
https://studentacademicsuccesssystem14-p-nosqllite.streamlit.app/
Or
With SqlLite
https://studentacademicsuccesssystem14-p-withsqllite.streamlit.app/
```
If you want to run by terminal or if there was some issues with link try to do steps below:

Firstly change the path into right form:
```bash
cd C:\Users\yourpath\Studen_Academic_Success_System_14-P\.venv\

in my case:
cd C:\Users\Омар\PycharmProjects\Studen_Academic_Success_System_14-P\.venv\
```

And you should fully change the paths of each models

```python
dt_model = joblib.load('yourpath/.venv/Trained models/decision_tree_model.joblib')
lr_model = joblib.load('yourpath/.venv/Trained models/logistic_regression_model.joblib')
rf_model = joblib.load('yourpath/.venv/Trained models/random_forest_model.joblib')
scaler = joblib.load('yourpath/.venv/Trained models/scaler.joblib')
columns = joblib.load('yourpath/.venv/Trained models/columns.joblib')  # Load the columns used during training
```

To run the app, use the following command:
```bash
streamlit run src/streamlitApp.py
```
**streamlitApp.py:** The main Streamlit app file that provides the front-end interface for users.

For a version that stores predictions in an SQLite database, run:

```bash
streamlit run src/streamlitAppSQLite.py
```

**streamlitAppSQLite.py:** A variant of the Streamlit app that uses SQLite to store user predictions.

## Training & Model Evaluation

### Training the Models
Training scripts are included in the ```Final-3.ipynb``` and ```FinalPM.ipynb``` Jupyter Notebooks. The models are trained with various hyperparameters and evaluated based on the following metrics:

**Accuracy:** Percentage of correct predictions.

**Precision:** Measure of correct positive predictions.

**Recall:** Measure of correct identification of positive cases.

**F1 Score:** Harmonic mean of precision and recall.

## Saving Models and Scaler
Once the models are trained, they are saved using ```joblib``` for later use. This includes saving the models, the scaler (used for feature scaling), and the column transformer for preprocessing.

You can load these saved models with the following:
```python
from joblib import load

# Load the trained models
dt_model = load('Trained models/decision_tree_model.joblib')
lr_model = load('Trained models/logistic_regression_model.joblib')
rf_model = load('Trained models/random_forest_model.joblib')

# Load the scaler
scaler = load('Trained models/scaler.joblib')
```
## Results
Once training is complete, the system outputs:

**Model Accuracy:** A comparison of each model's accuracy.

**Classification Reports:** Detailed metrics for each model (precision, recall, F1-score).

**User Prediction:** The system predicts whether a student is likely to succeed or fail academically based on input data.

## Acknowledgements
The dataset used in this project is given by SDU teachers.

This project uses machine learning algorithms from scikit-learn.

The Streamlit library is used to create an interactive web app for making predictions.

Special thanks to the SDU University community for providing valuable resources.

**And special thanks for Bakdaulet and Sufyan teachers for this great semestr and we wish all the best for both teachers:)**

## References
“Artificial intelligence modern Application”-Stuart Russel, Peter Norvig

https://youtube.com/playlist?list=PLa6CNrvKM5QU7AjAS90zCMIwi9RTFNIIW&feature=shared 

https://support.zyte.com/support/solutions/articles/22000201935-deploying-a-project-from-a-github-repository

https://habr.com/ru/articles/510934/

https://scikit-learn.org/stable/ 

https://habr.com/ru/articles/864890/

https://www.datacamp.com/tutorial/random-forests-classifier-python

https://asana.com/ru/resources/decision-tree-analysis

https://habr.com/ru/companies/skillfactory/articles/675876/

https://medium.com/@thomas.vidori/how-to-create-a-python-package-and-publish-it-on-github-eebc78b2a12d

## Team Members

**Omar Aitimbet** with ID: **220103131**

**Dinmukhamed Sapybek** with ID: **220103053**

