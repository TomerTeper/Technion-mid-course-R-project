
# Red Wine Quality Analysis

This project analyzes the quality of red wine using various machine learning techniques. The dataset used is from the UCI Machine Learning Repository Kaggle.

## Table of Contents

1. [Dataset](#dataset)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling](#modeling)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
   - [Elastic Net](#elastic-net)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)

## Dataset

The dataset used in this project can be found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). It contains various chemical properties of red wine along with a quality rating.

## Data Preparation

1. Load the dataset.
2. Transform the quality variable into a binary variable:
   - Quality > 6.5 is considered good (1).
   - Quality ≤ 6.5 is considered bad (0).
3. Check for and handle missing values.

## Exploratory Data Analysis (EDA)

- Visualize the distribution of different features using plots.
- Create a heatmap to understand the correlation between features.

## Modeling

### K-Nearest Neighbors (KNN)

1. Normalize the numeric variables.
2. Split the data into training (70%) and testing (30%) sets.
3. Implement KNN with k values of 1, 3, and 5.
4. Evaluate the models using confusion matrices and classification accuracy.

### Linear Regression

1. Fit a linear regression model on the training set.
2. Evaluate the model using Mean Squared Error (MSE) on the test set.

### Logistic Regression

1. Fit a logistic regression model.
2. Evaluate the model using ROC curves and confusion matrices.
3. Calculate sensitivity, specificity, precision, and negative predictive value.

### Decision Tree

1. Split the data into training (50%) and testing (50%) sets.
2. Fit a decision tree model.
3. Prune the tree to optimize performance.
4. Evaluate the model using MSE.

### Random Forest

1. Split the data into training (70%) and testing (30%) sets.
2. Fit a random forest model with 1500 trees.
3. Determine the optimal number of trees and evaluate the feature importance.
4. Perform cross-validation to optimize the model.
5. Evaluate the model using MSE.

### Elastic Net

1. Split the data into training (70%) and testing (30%) sets.
2. Use cross-validation to find the best alpha and lambda values.
3. Fit an elastic net model with the optimal parameters.
4. Evaluate the model using MSE.

## Installation

1. Clone the repository:
   \`\`\`sh
   git clone https://github.com/your-username/red-wine-quality-analysis.git
   \`\`\`
2. Navigate to the project directory:
   \`\`\`sh
   cd red-wine-quality-analysis
   \`\`\`
3. Install the required R packages:
   \`\`\`R
   install.packages(c("ggplot2", "reshape2", "readxl", "class", "InformationValue", "mccr", "ROCit", "ISLR", "pastecs", "randomForest", "tree", "MASS", "leaps", "glmnet"))
   \`\`\`

## Usage

1. Load the R script:
   \`\`\`R
   source("red_wine_quality_analysis.R")
   \`\`\`
2. Follow the steps in the script to preprocess the data, perform EDA, and build models.

## Results

The models are evaluated based on different metrics such as accuracy, MSE, and ROC curves. The random forest model showed the best performance with an MSE of 0.06032756.

