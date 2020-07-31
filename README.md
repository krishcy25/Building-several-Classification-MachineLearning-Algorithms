# Building-several-Classification-MachineLearning-Algorithms

![cl](https://user-images.githubusercontent.com/65406908/89031078-c57cb800-d2ff-11ea-8a24-059ebcd742d4.jpg)

This repository focuces on building several Classification Machine Learning Models that predicts the decision of the loan approval based on customer attributes. I have worked on this project as part of Analytics Vidhya Hackathon

Although, I have a repository earlier with the same Hackathon- with PyCaret (using few lines of code)- This respository focuses on building several Classification models from scratch following the traditional Data Mining Approach of working on data preparation, data exploration, model buidling and validation

This repository includes the code for the below:


# Step 1: Data Exploration

Technique 0.1: Pandas Profiling- A single step data exploratory technique

Technique 0.2: Checking the train data set to get feel of the dataset

Technique 0.3: To research on the variables (thier type and number of missing values)

Technique 0.4: Analyzing and researching the target variable for outliers, skewness

Technique 0.5- Analzying more about correlated features

Technique 0.6- Analyzing the top features that has close relationship/strong correlation with target. The top features are then checked for outlier removal (if needed)

Technique 0.7- Ground work for making data ready for data preparation

# Step 2: Data Preparation

Technique 1: One Hot Encoding

Technique 2: Handling Missing data (Using Simple Random Imputation and Deterministic Imputation)

Technique 3: Feature Engineering

Technique 4: Removing correlated features
      
      #4.1 Handling of constant columns with Variance Threshold
      
      #4.2 Removing Quasi Constant features
      
      #4.3 Removing duplicate features
      
      #4.4 Removing correlated features using corr
  
Technique 5: Transformations and Scaling


# Step 3: Data Splitting

Data Splitting  Step 1: Dividing the train vs test separately after data preparation (as we combined them at the start before data preparation)

Data Splitting  Step 2: Splitting into train vs validation (using train_test_split)


# Step 4: Model Building

Model 1: Logistic Regression (that predicts the output in the form of probabilities between 0 and 1). Here probabilities refer to the chance of loan approval decision

Model 1.2: Logistic Regression that predicts output in the form of 0/1's as the Hackathon requires us to submit predictions in the form of Y/N

Model 2: XG Boost with Hyperparameters

Model 3: KNN Classifier

Model 4: Support Vector Machines (SVC)

Model 5:  Naive Bayes Classifier



All the data files required to train/validate the model can be found in the data folder with in same repository and all the submission files that I have submitted for the Hackathon were included in the same repository under "submission_files" folder.

Hackathon link can be found under 

https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

![bk](https://user-images.githubusercontent.com/65406908/89031105-d62d2e00-d2ff-11ea-99a8-2456022547b7.png)

