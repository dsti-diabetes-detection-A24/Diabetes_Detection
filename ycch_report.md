
# Yohann's job on the projects
## Exploratory Data Analysis (EDA)

### Data Understanding and Quality Assessment

The diabetes detection project involved building a predictive model to identify diabetic patients using various health metrics. The process began with thorough data exploration of the TAIPEI diabetes dataset containing approximately 15,000 records across multiple features.

After importing necessary libraries and loading the dataset, I performed a comprehensive exploratory analysis. The initial statistical summary and correlation analysis revealed potentially suspicious patterns - the dataset appeared mathematically "too perfect." This observation led me to investigate deeper and cross-analyze relationships between variables.

Given that this is a medical dataset from a hospital environment, some extreme values might be expected. However, distinguishing between legitimate clinical outliers and data errors required domain knowledge. I systematically identified potential inconsistencies by defining logical rules based on medical principles and human physiology.

### Inconsistency Detection

I identified several problematic patterns in the dataset and we worked together to implement rules:

1. **Age-Pregnancy Relationship**: Young women (≤25 years) with unusually high pregnancy counts (>5)
2. **Implausible BMI Values**: Extreme BMI measurements outside realistic physiological ranges
3. **Glucose-Insulin Relationship**: High plasma glucose (>200 mg/dL) with unexpectedly low insulin levels
4. **Blood Pressure Anomalies**: Diastolic blood pressure readings below 40 mmHg or above 120 mmHg
5. **Extreme Insulin Values**: Serum insulin measurements outside plausible clinical ranges

### Code overview of cross-Validation with PIMA Dataset

The TAIPEI dataset was compared with the established PIMA diabetes dataset from Kaggle to validate outlier concerns and ensure consistency across similar diabetes studies. The PIMA dataset serves as a reliable benchmark as it:

- Comes from the National Institute of Diabetes and Digestive and Kidney Diseases
- Has been widely used in diabetes research
- Contains similar medical predictor variables
- Has undergone rigorous validation

The comparison confirmed that several observations in our dataset were indeed outliers beyond typical clinical variation. After consulting with instructor Dr. Christophe Bécavin, I proceeded with data cleaning based on the identified inconsistencies.

#### Comparison Between TAIPEI and PIMA Diabetes Datasets

##### 1. Dataset Loading and Column Alignment

```python
# Import PIMA dataset
pima_diabetes = pd.read_csv('ressources/kaggle/diabetes.csv')

# Rename columns to match between datasets for comparison
diabetes_csv_renamed = diabetes_csv.rename(columns={
    'Diabetic': 'Outcome',
    'DiastolicBloodPressure': 'BloodPressure',
    'DiabetesPedigree': 'DiabetesPedigreeFunction',
    'PlasmaGlucose': 'Glucose',
    'SerumInsulin': 'Insulin',
    'TricepsThickness': 'SkinThickness'
})
```

The first step involved aligning column names between the two datasets to ensure valid comparisons, as they had different naming conventions.

##### 2. Statistical Comparison

The notebook calculates and compares basic statistical properties of both datasets:

```python
# Calculate min and max for both datasets
pima_min_max = pima_diabetes.agg(['min', 'max']).T
diabetes_min_max = diabetes_csv.agg(['min', 'max']).T

# Join for comparison
comparison = pima_min_max.join(diabetes_min_max)
```

This comparison revealed differences in value ranges between the datasets, helping identify potential anomalies in the TAIPEI dataset.

##### 3. Outlier Validation

For each suspected outlier in the TAIPEI dataset, the notebook examines equivalent cases in the PIMA dataset to determine if they're truly anomalous or within expected ranges:

```python
# Row of max pregnancies for young individuals
pima_diabetes[(pima_diabetes['Pregnancies'] == pima_diabetes['Pregnancies'].max()) & (pima_diabetes['Age'] < 25)]

# Row of plasma glucose above 160
pima_diabetes[pima_diabetes['Glucose'] > 160].sort_values(by='Glucose', ascending=False)

# Various other outlier checks for blood pressure, BMI, insulin, etc.
```

This process allowed for identification of truly suspect values by comparing against a reference dataset from a similar population.

##### 4. Distribution Visualization

Side-by-side histograms were created to compare feature distributions between datasets:

```python
# Define age bins and create two-panel histogram comparison
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)
axes[0].hist(diabetes_csv['Age'], bins=bins, edgecolor='black')
axes[1].hist(pima_diabetes['Age'], bins=bins, edgecolor='black')
```

These visualizations helped identify structural differences between datasets, such as different age distributions and feature value concentrations.

##### 5. Model Comparison

The notebook builds identical logistic regression models on both datasets to compare predictive performance:

```python
# Train separate models for each dataset
model_pima = LogisticRegression()
model_pima.fit(X_pima_train, y_pima_train)

model_diabetes = LogisticRegression()
model_diabetes.fit(X_diabetes_train, y_diabetes_train)

# Compare evaluation metrics
```

This model-based comparison revealed how data quality and distribution differences affect prediction capabilities.

##### 6. Testing Data Cleaning Impact

The code tests how removing specific outliers affects model performance:

```python
# Remove rows with anomalous values
diabetes_csv_no_max_pregnancies = diabetes_csv.drop(diabetes_csv[(diabetes_csv['Pregnancies'] == diabetes_csv['Pregnancies'].max()) & (diabetes_csv['Age'] < 25)].index)

# Build models on cleaned data to measure performance differences
```

This systematic comparison iteration helped validate data quality and determine whether unusual values are genuine outliers or natural variations in the dataset.

### Data Cleaning and Preparation

Rather than simply removing outliers, I applied a combination of techniques:
- Identifying and removing biologically impossible combinations of values
- Capping extreme values at the 8th and 92nd percentiles to preserve distribution shapes
- Verifying data completeness after cleaning operations

The cleaned dataset maintained its integrity while removing problematic observations that could potentially skew modeling results.

## Feature Engineering

After data cleaning, I enhanced the dataset **in my branch of the project** with meaningful derived features to capture complex physiological relationships:

### Age Group Categorization
I transformed the continuous age variable into clinically relevant age categories:
- 20-30 years
- 31-40 years
- 41-50 years
- 51-60 years
- 61-70 years
- 71-80 years

This transformation acknowledges that diabetes risk doesn't increase linearly with age but follows distinct patterns across different life stages. For example, type 2 diabetes risk significantly accelerates after age 45, and these categories enable the model to capture such non-linear relationships.

### BMI Classification
I converted BMI from a continuous measurement into standardized clinical categories according to WHO guidelines:
- Underweight (<18.5)
- Normal weight (18.5-24.9)
- Overweight (25-29.9)
- Obese Class I (30-34.9)
- Obese Class II (35-39.9)
- Obese Class III (≥40)

This transformation incorporates established medical knowledge about risk thresholds, as diabetes risk increases disproportionately at specific BMI levels rather than continuously.

### Glucose-Insulin Interaction
I created an interaction term between plasma glucose and serum insulin to capture their physiological relationship, as insulin resistance (characterized by high glucose despite high insulin) is a key diabetes indicator.

### Metabolic Risk Score
I developed a composite score combining BMI, blood pressure, and glucose values to create a holistic metabolic risk indicator. This feature captures the cumulative effect of multiple risk factors that might individually fall below diagnostic thresholds.

### Feature Encoding and Transformation
For implementation in machine learning models:
- Categorical features were one-hot encoded
- Skewed numerical features underwent log transformation to normalize their distributions
- All features were standardized using MinMaxScaler to ensure equal contribution to the model

Additionally, I applied SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance between diabetic and non-diabetic samples, creating a balanced dataset for more effective modeling.

### Impact on Model Performance
These feature engineering techniques substantially improved model performance:
- Accuracy increased from 79% to 84%
- Recall for diabetic cases (sensitivity) improved from 56% to 69%
- Precision for diabetic predictions increased from 73% to 80%
- F1-score improved from 63% to 74%

Most notably, the age group and BMI category features demonstrated high importance in the final model, confirming that the categorical transformation of these variables effectively captured critical threshold effects that would have been less apparent with continuous variables.

These preprocessing steps and feature engineering techniques significantly improved the dataset quality and prepared it for the subsequent modeling phase.

## Logistic Regression Model for Diabetes Detection

### Team Approach to Modeling

As part of our comprehensive approach to the Diabetes Detection project, our team decided to evaluate multiple machine learning models to identify the most effective one for predicting diabetes. Each team member was assigned a different model to implement, evaluate, and present. I was specifically responsible for developing and analyzing the **logistic regression** model as my contribution to our comparative study.

### Why Logistic Regression?

Logistic regression was selected as a baseline model for several key reasons:

1. **Binary Classification Problem**: Since we're predicting whether a patient has diabetes (1) or not (0), logistic regression is specifically designed for such binary outcomes, unlike linear regression which predicts continuous values.

2. **Interpretability**: Unlike "black box" models, logistic regression provides clear coefficients that can be interpreted as the log-odds impact of each feature, which is particularly valuable in healthcare applications where understanding feature importance is crucial.

3. **Probabilistic Output**: Rather than simply classifying patients, logistic regression provides probability estimates of having diabetes, allowing for threshold adjustments based on sensitivity/specificity requirements.

4. **Medical Research Standard**: Logistic regression has a long history of use in epidemiological studies and medical research, making results more accessible to healthcare professionals.

### Implementation Process

The implementation of the logistic regression model followed a structured approach:

1. **Feature Engineering**: Created meaningful derived features from raw data:
   - Age groups (20-30, 31-40, 41-50, 51-60, 61-70, 71-80) to capture non-linear age effects
   - BMI categories (Underweight, Normal, Overweight, Obese) based on standard medical classifications
   - One-hot encoded these categorical variables to make them suitable for the model

2. **Data Preprocessing**:
   - Standardized numerical features using StandardScaler to ensure equal contribution from all variables
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance between diabetic and non-diabetic patients

3. **Model Training and Evaluation**:
   - Split the data into training (80%) and testing (20%) sets
   - Fitted the logistic regression model and evaluated using accuracy, confusion matrix, and classification report
   - Analyzed feature coefficients to understand the impact of each variable on diabetes prediction

### Advantages of Logistic Regression

Our implementation highlighted several advantages of logistic regression:

- **Computational Efficiency**: The model trained quickly even on our large dataset of 15,000 records
- **Feature Importance Analysis**: We could directly interpret which factors most strongly influence diabetes risk
- **Balanced Performance**: The model achieved respectable accuracy (~78%) while maintaining balance between precision and recall
- **Simplicity**: The straightforward implementation allowed us to focus on feature engineering rather than complex model tuning

### Limitations of Logistic Regression

Despite its strengths, we also recognized several limitations:

- **Linear Decision Boundary**: Logistic regression assumes a linear relationship between features and log-odds, potentially missing complex interactions between variables
- **Limited Capacity for Non-Linear Relationships**: Despite our feature engineering efforts to capture non-linearity through categorical variables, more complex relationships might still be missed
- **Sensitivity to Outliers**: Our data exploration revealed several extreme values (particularly in insulin levels) that could disproportionately influence the model
- **Assumption of Feature Independence**: The model doesn't naturally account for correlations between predictors like BMI and blood pressure

## Contribution to Final Model Selection

This logistic regression implementation served as an excellent baseline model against which other more complex models (random forests, gradient boosting, neural networks) could be compared. The interpretability of its coefficients provided valuable insights about feature importance that informed feature selection in other models.

The final team decision on which model to adopt balanced predictive performance with interpretability requirements, computational efficiency, and other project-specific considerations across all the models evaluated by team members.