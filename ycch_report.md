# Project Overview

## Goal
The primary goal of this project is to build a machine learning model to predict diabetic outcomes using the provided dataset. The model will be evaluated and compared with the PIMA dataset to ensure the coherence and reliability of our dataset.

## Steps Involved
1. **Import Required Libraries**: Import necessary libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
2. **Load the Dataset**: Load the dataset into a pandas DataFrame for analysis.
3. **Data Preprocessing**: Handle missing values, detect and handle outliers, and perform feature scaling.
4. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the distribution and relationships of the features.
5. **Comparison with PIMA Dataset**: Compare the dataset with the PIMA dataset to verify the coherence of our dataset.
6. **Modeling**: Split the data into training and testing sets, train a machine learning model, and evaluate its performance.
7. **Feature Engineering**: Create new features to improve the model's performance.
8. **Model Evaluation**: Evaluate the model using various metrics and interpret the results.
9. **Improvements**: Look for potential improvements in the model and the dataset.

# Data Collection and Import

## Explanation
In this section, we will explain the process of collecting and importing the dataset. We will use the pandas library to read the CSV file containing the dataset. The dataset provides attributes for 15000 women on 8 features, and the variable to predict is the diabetic outcome.

## Code
```python
# Import required libraries
import pandas as pd

# Load the dataset into a pandas DataFrame
diabetes_csv = pd.read_csv('data/TAIPEI_diabetes.csv')

# Display the first few rows of the dataset to verify the import
diabetes_csv.head()
```

# Exploratory Data Analysis (EDA)

## Discuss the steps taken to explore the data, including checking for missing values, visualizing distributions, and identifying potential outliers.

### Checking for Missing Values
```python
# Check for missing values in the dataset
missing_values = diabetes_csv.isna().sum()
missing_values
```

### Visualizing Distributions
```python
# Visualize the distribution of each feature using histograms
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_csv.hist(figsize=(20, 10), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()
```

### Identifying Potential Outliers
```python
# Use box plots to identify potential outliers in the numerical features
numeric_cols = diabetes_csv.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=diabetes_csv[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# Calculate the IQR to identify outliers
Q1 = diabetes_csv.quantile(0.25)
Q3 = diabetes_csv.quantile(0.75)
IQR = Q3 - Q1

outliers = ((diabetes_csv < (Q1 - 1.5 * IQR)) | (diabetes_csv > (Q3 + 1.5 * IQR))).sum()
outliers

# Calculate Z-scores to identify outliers
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(diabetes_csv.select_dtypes(include=[np.number])))
outliers_z = (z_scores > 3).sum(axis=0)
outliers_z
```

## Comparison with PIMA Dataset
In my EDA, I noticed some incoherence and excessive values like outliers but I wasn't sure, so I decided to compare our "describe" values to another dataset, the PIMA dataset, as a reference. Then I suggested to the group to remove some values.

# Modeling

## Explanation
In this section, we will detail the modeling process, including the choice of logistic regression, data preprocessing, splitting the data into training and testing sets, and training the model.

### Choice of Logistic Regression
Logistic regression is chosen for this project because it is a simple yet effective algorithm for binary classification problems. It provides probabilities for class membership and is easy to interpret.

### Data Preprocessing
We will preprocess the data by scaling the features to ensure that they are on the same scale. This helps improve the performance of the logistic regression model.

### Splitting the Data
We will split the data into training and testing sets to evaluate the model's performance on unseen data.

### Training the Model
We will train a logistic regression model using the training data and evaluate its performance using the testing data.

## Code
```python
# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Drop unnecessary columns
diabetes_csv = diabetes_csv.drop(columns=['PatientID'], errors='ignore')

# Separate features and target variable
X = diabetes_csv.drop(columns=['Diabetic'])
y = diabetes_csv['Diabetic']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the evaluation results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```

## Handling Outliers in Modeling
In my modeling, I tried to remove different possible outliers to see how it affects my model. This involved iterating through the data, removing outliers, and retraining the model to observe changes in performance.

# Evaluation and Comparison with PIMA Dataset

## Discuss the evaluation metrics used to assess the model's performance, including accuracy, precision, recall, and F1 score.

### Evaluation Metrics
In this section, we will discuss the evaluation metrics used to assess the performance of our logistic regression model. The metrics include accuracy, precision, recall, and F1 score.

- **Accuracy**: The proportion of correctly classified instances among the total instances.
- **Precision**: The proportion of true positive instances among the instances predicted as positive.
- **Recall**: The proportion of true positive instances among the actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

### Model Evaluation Results
The evaluation results for our logistic regression model are as follows:

- **Accuracy**: {accuracy}
- **Confusion Matrix**:

# Conclusion

## Summary of Findings
In this project, we aimed to build a machine learning model to predict diabetic outcomes using the provided dataset. We followed a structured approach, including data preprocessing, exploratory data analysis, feature engineering, modeling, and evaluation. The key findings and results are summarized below:

### Data Preprocessing and EDA
- We handled missing values and identified potential outliers using both the IQR and Z-score methods.
- Visualizations such as histograms and box plots helped us understand the distribution of features and detect outliers.

### Feature Engineering
- We created new features such as age groups and BMI categories to improve the model's performance.
- One-hot encoding was applied to convert categorical variables into numerical format.

### Modeling
- We chose logistic regression for its simplicity and effectiveness in binary classification problems.
- The data was split into training and testing sets, and features were scaled to ensure uniformity.
- The logistic regression model was trained and evaluated using various metrics.

### Model Evaluation
- The model achieved an accuracy of {accuracy}, with the following confusion matrix and classification report:
```



1782208