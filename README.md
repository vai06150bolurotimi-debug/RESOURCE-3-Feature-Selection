RESOURCE_3 — Feature Selection with the Wine Dataset

Overview

This repository contains the replication of RESOURCE_3, a practical exercise focused on feature selection in machine learning.

The notebook uses the Wine Recognition dataset available through Scikit-learn and a Gradient Boosting Classifier to investigate how different feature-selection techniques affect model performance.

The resource compares several approaches:

Variance Threshold feature selection

SelectKBest using Mutual Information

Recursive Feature Elimination (RFE)

Boruta feature selection

The performance of the different approaches is evaluated using the weighted F1-score.

Learning Objectives

The notebook is designed to demonstrate how to:

Load and explore a machine-learning dataset.

Separate input features from the target variable.

Split data into training and testing sets.

Build a baseline Gradient Boosting classification model.

Evaluate model performance using weighted F1-score.

Investigate feature variance.

Perform variance-based feature selection.

Select features using Mutual Information and SelectKBest.

Perform model-based feature selection using RFE.

Perform feature selection using Boruta.

Identify the actual features selected by different methods.

Compare the performance of the feature-selection approaches visually.

Dataset

The notebook uses Scikit-learn's built-in Wine Recognition dataset through:

from sklearn.datasets import load_wine

The dataset contains:

178 observations

13 input features

3 target classes

The target variable represents the wine category/class.

The dataset is loaded and converted into a Pandas DataFrame:

wine_data = load_wine()

wine_df = pd.DataFrame(
    data=wine_data.data,
    columns=wine_data.feature_names
)

wine_df['target'] = wine_data.target

Features

The Wine dataset contains 13 input features describing characteristics of the wine.

The target column is kept separate from the input features before machine-learning training:

X = wine_df.drop(['target'], axis=1)
y = wine_df['target']

Here:

X represents the input features.

y represents the target variable.

Train/Test Split

The data is divided into training and testing sets using train_test_split:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=True,
    stratify=y,
    random_state=42
)

The notebook uses:

70% training data

30% testing data

Shuffling

Stratification based on the target

random_state=42 for reproducibility

Baseline Model

Before applying feature selection, the notebook creates a baseline model using all available features.

The model is:

GradientBoostingClassifier(
    max_depth=5,
    random_state=42
)

The model is trained using:

gbc.fit(X_train, y_train)

Predictions are generated using:

preds = gbc.predict(X_test)

The model is evaluated using weighted F1-score:

f1_score_all = round(
    f1_score(y_test, preds, average='weighted'),
    3
)

The baseline provides a reference point for comparing the different feature-selection techniques.

Feature Selection Methods

1. Variance-Based Feature Selection

The first approach investigates the variance of the features.

Variance measures how much the values of a feature differ from one another.

The notebook calculates feature variance using:

X_train_v1.var(axis=0)

The data is also scaled using MinMaxScaler before visualizing the variance:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_X_train_v1 = scaler.fit_transform(X_train_v1)

The resource identifies ash and magnesium for removal based on the variance threshold used in the exercise.

The selected data is then used to retrain the Gradient Boosting Classifier and calculate another weighted F1-score.

2. SelectKBest with Mutual Information

The second approach is a filter method.

The notebook uses:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

Mutual Information is used to assess how informative individual features are about the target.

The notebook tests different numbers of selected features:

for k in range(1, 14):

For each value of k, the process is:

Create a SelectKBest selector.

Fit it using the training data.

Transform the training data.

Transform the test data using the same selector.

Train the Gradient Boosting Classifier.

Make predictions.

Calculate weighted F1-score.

Store the result.

The scores are stored in:

f1_score_list

The notebook then plots F1-score against the number of selected features.

Checking the Selected Features

The notebook also demonstrates how to identify the actual features selected when k=3:

selector = SelectKBest(mutual_info_classif, k=3)

selector.fit(X_train_v2, y_train_v2)

selected_feature_mask = selector.get_support()

selected_features = X_train_v2.columns[
    selected_feature_mask
]

selected_features

get_support() produces a Boolean selection mask, and that mask is applied to the DataFrame's column names to obtain the selected feature names.

3. Recursive Feature Elimination (RFE)

The third approach is presented in the notebook as a wrapper method.

RFE stands for Recursive Feature Elimination.

Unlike the Mutual Information approach, RFE uses the machine-learning model itself to determine feature importance.

The notebook uses:

from sklearn.feature_selection import RFE

The Gradient Boosting Classifier is supplied as the estimator:

RFE_selector = RFE(
    estimator=gbc,
    n_features_to_select=k,
    step=1
)

The notebook tests different values of k, from 1 to 13.

For each value of k, RFE:

Uses the Gradient Boosting Classifier as the estimator.

Selects the requested number of features.

Transforms the training data.

Transforms the test data.

Retrains the classifier.

Generates predictions.

Calculates the weighted F1-score.

Stores the result.

The results are stored in:

rfe_f1_score_list

A graph is then used to compare the F1-score obtained with different numbers of selected features.

4. Boruta Feature Selection

The final feature-selection technique demonstrated is Boruta.

The notebook installs the package with:

!pip install Boruta

and imports:

from boruta import BorutaPy

Boruta is then initialized using the Gradient Boosting Classifier:

boruta_selector = BorutaPy(
    gbc,
    random_state=42
)

The selector is fitted using the training data:

boruta_selector.fit(
    X_train_v4.values,
    y_train_v4.values.ravel()
)

The selected training and testing features are obtained with:

sel_X_train_v4 = boruta_selector.transform(
    X_train_v4.values
)

sel_X_test_v4 = boruta_selector.transform(
    X_test_v4.values
)

The classifier is then retrained using the Boruta-selected features and evaluated:

gbc.fit(sel_X_train_v4, y_train_v4)

boruta_preds = gbc.predict(sel_X_test_v4)

boruta_f1_score = round(
    f1_score(
        y_test_v4,
        boruta_preds,
        average='weighted'
    ),
    3
)

The notebook uses the test labels, y_test_v4, when calculating the final Boruta F1-score.

Checking Boruta's Selected Features

The actual selected features can be displayed with:

selected_features_mask = boruta_selector.support_

selected_features = X_train_v4.columns[
    selected_features_mask
]

selected_features

Here:

support_ provides Boruta's Boolean selection mask.

True indicates that a feature was selected.

False indicates that it was not selected.

Applying the mask to X_train_v4.columns returns the feature names.

Model Evaluation

The notebook uses the weighted F1-score to evaluate model performance.

The general evaluation pattern is:

f1_score(
    y_test,
    predictions,
    average='weighted'
)

The F1-score combines precision and recall into a single performance measure.

The notebook notes that, generally, a score closer to 1 indicates better model performance.

The different F1-score variables include:

f1_score_all
f1_score_var
f1_score_kbest
f1_score_rfe
boruta_f1_score

Final Comparison

The notebook concludes by comparing the feature-selection methods in one bar chart:

x = [
    'All features (13)',
    'Variance threshold (11)',
    'Filter - MI (3)',
    'RFE(3)',
    'Boruta (9)'
]

y = [
    f1_score_all,
    f1_score_var,
    0.981,
    1.0,
    boruta_f1_score
]

The chart compares:

Approach

Features Used

All features

13

Variance threshold

11

Filter - Mutual Information

3

RFE

3

Boruta

9

The purpose of this final visualization is to make it easier to compare model performance after different feature-selection approaches.

Technologies and Libraries

The notebook uses the following Python libraries and tools:

Python

The programming language used for the machine-learning workflow.

Pandas

Used for:

Creating DataFrames

Loading and manipulating tabular data

Selecting and dropping columns

Inspecting the dataset

NumPy

Used for numerical operations and creating ranges for plotting.

Matplotlib

Used to create visualizations such as:

Feature variance charts

F1-score comparison charts

Feature-selection performance plots

Seaborn

Used for the swarm plot that visualizes selected feature distributions across target classes.

Scikit-learn

Used for:

Loading the Wine dataset

Train/test splitting

Gradient Boosting Classification

F1-score evaluation

MinMax scaling

SelectKBest

Mutual Information

RFE

Boruta

Used for Boruta-based feature selection.

Google Colab

The notebook contains a Google Colab link and can be run in a Colab environment.

Repository Structure

A recommended repository structure is:

Student-resource-2-replication/
│
├── RESOURCE_3.ipynb
├── README.md
└── ...

RESOURCE_3.ipynb

The main Jupyter Notebook containing the complete practical implementation.

README.md

This document. It explains the project, notebook contents, techniques, tools, and key learning points.

Additional documentation, screenshots, reports, or supporting files can be added to the repository if required by the assignment.

How to Run the Notebook

Option 1: Google Colab

Open the notebook in Google Colab using the Colab link included at the beginning of the notebook.

The notebook already contains:

%matplotlib inline

and installs Boruta with:

!pip install Boruta

when required.

Option 2: Jupyter Notebook / JupyterLab

Clone or download the repository and open:

RESOURCE_3.ipynb

Install the required packages if they are not already installed:

pip install pandas numpy matplotlib seaborn scikit-learn Boruta

Then run the notebook cells from top to bottom.

Recommended Execution Order

The notebook is structured as a progressive practical:

Load libraries
      ↓
Load Wine dataset
      ↓
Explore the dataset
      ↓
Create training/testing sets
      ↓
Build baseline Gradient Boosting model
      ↓
Calculate baseline F1-score
      ↓
Variance-based feature selection
      ↓
Evaluate variance-selected model
      ↓
SelectKBest + Mutual Information
      ↓
Evaluate different numbers of features
      ↓
Identify selected features
      ↓
RFE
      ↓
Evaluate different numbers of features
      ↓
Boruta
      ↓
Evaluate Boruta-selected features
      ↓
Identify Boruta-selected features
      ↓
Compare all approaches

Key Takeaways

The main lesson from this resource is that feature selection can be used to reduce the number of input variables supplied to a machine-learning model while evaluating how this affects predictive performance.

The notebook demonstrates that different feature-selection methods approach the problem differently.

Variance-based selection looks at how much individual features vary.

Mutual Information with SelectKBest ranks individual features according to their information about the target and allows a specified number of top features to be selected.

RFE uses the machine-learning estimator to recursively eliminate less important features.

Boruta provides another model-based feature-selection approach and is used to determine relevant features before the classifier is retrained.

The final F1-score comparison allows the different approaches to be evaluated against the baseline model that uses all 13 features.

Important Concepts

The main concepts covered in this resource are:

Feature selection

Training and testing data

Classification

Gradient Boosting

Weighted F1-score

Feature variance

Variance-based selection

Filter methods

Mutual Information

SelectKBest

Wrapper methods

Recursive Feature Elimination

Boruta

Boolean feature-selection masks

Model performance comparison

Data visualization

Project Outcome

By completing this resource, the learner gains practical experience in applying multiple feature-selection techniques to a classification problem and comparing their effects on model performance.

The notebook also demonstrates an important machine-learning workflow:

Dataset
   ↓
Exploration
   ↓
Train/Test Split
   ↓
Baseline Model
   ↓
Feature Selection
   ↓
Model Training
   ↓
Prediction
   ↓
F1-Score Evaluation
   ↓
Performance Comparison

Author

Olurotimi Favour

Vephla University

Course/Project: Artificial Intelligence / Machine Learning

Note

This README is based on the contents and terminology used in RESOURCE_3.ipynb. It is intended to document the notebook and make the repository easier to navigate and understand.
