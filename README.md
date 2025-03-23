# Diabetes Detection Project

A machine learning-based web application for predicting diabetes risk in women based on health metrics.

## Overview

This project uses several machine learning algorithms to predict the likelihood of diabetes in women based on key health indicators. The models are trained on a dataset containing health metrics and have been optimized for accuracy and performance.

## Features

- Multiple trained machine learning models:
  - Random Forest
  - Logistic Regression
  - AdaBoost
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
- Interactive web interface for predictions
- Comprehensive model evaluation and comparison
- Data visualization and analysis

## Dataset

The project uses the Pima Indians Diabetes Dataset, containing these features:
- Number of pregnancies
- Plasma glucose concentration
- Diastolic blood pressure
- Triceps skin fold thickness
- 2-Hour serum insulin
- Body mass index (BMI)
- Diabetes pedigree function
- Age

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/Diabetes_Detection.git cd Diabetes_Detection
```
2. Create and activate a virtual environment (recommended):`

```
python -m venv env
```
On Windows

```
env\Scripts\activate
```

On macOS/Linux

```
source env/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt

```

## Usage

### Running the Web Application

To start the web application:

```
python app.py

```


The application will be accessible at http://localhost:7860 in your web browser.

### Using the Application

1. Enter the patient's health metrics in the provided fields
2. Select the preferred model type from the dropdown menu
3. Click the "Predict" button to get the diabetes risk assessment
4. Use the "Clear" button to reset all fields

## Project Structure

- `app.py` - The Gradio web application
- `*.pkl` - Trained model files
- Jupyter Notebooks:
  - `Notebook.ipynb` - Main project notebook with model comparison
  - `MLProject - DEA.ipynb` - Data exploration and analysis
  - `MLProject - KNN Algorithm.ipynb` - KNN model implementation
  - `MLProject_Decision_Tree_A.ipynb` - Decision tree model implementation
  - `logistic_regression.ipynb` - Logistic regression model
  - `random forest.ipynb` - Random forest implementation
  - `svm_model.ipynb` - SVM model implementation

## Model Performance

The models achieve accuracy scores of ~85-90% with different strengths:
- Random Forest: Best overall performance with balanced precision and recall
- Logistic Regression: Good interpretability and baseline performance
- AdaBoost: Strong performance on difficult cases
- SVM: Effective decision boundary for this classification task
- KNN: Simple but effective for this dataset
- Decision Tree: Good interpretability and visualization capabilities

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Gradio

## Future Improvements

- Add feature importance visualization
- Implement ensemble methods combining multiple models
- Add user authentication and result storage
- Deploy the application to a cloud platform

## License

This project is licensed under the MIT License - see the LICENSE file for details.
