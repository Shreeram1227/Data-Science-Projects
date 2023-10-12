
# Project 1: Per capita income prediction

### Description
This project aims to use linear regression to predict the per capita income of Canada in 2020 based on the historical data from 1970 to 2016. The data is obtained from a CSV file that contains the year and the corresponding per capita income in Canadian dollars

### Techniques Used
-Importing the necessary libraries such as pandas, numpy, sklearn, matplotlib, pickle and joblib.
-Loading the data from the csv file into a pandas dataframe.

-Exploring the data by plotting a scatter plot of year versus per capita income.

-Splitting the data into training and testing sets.

-Creating and fitting a linear regression model using sklearn.

-Data Visualization by using matplotlib

-Evaluating the model performance by calculating the mean absolute error, mean squared error and root mean squared error.

-Predicting the per capita income of Canada in 2020 using the model.
Saving the model using pickle and joblib methods for future use.

### Files
- `canada_per_capita_income.csv`: The dataset used for training and testing the models.
- `Per_capita_income_canada`: Jupyter Notebook containing the linear regression implementation.


  # Project 2: Predicting Salary of New Hires by using Multiple Linear Regression

  ### Description
This project uses multiple linear regression to predict the salary of new hires based on their experience, test scores and interview scores. The data is obtained from a CSV file that contains the following columns: experience, test_score, interview_score and salary.

### Techniques Used
Importing the necessary libraries such as pandas, numpy, sklearn and word2number.
Loading the data from the csv file into a pandas dataframe.
Converting the experience column from words to numbers using word2number library.
Handling any missing values in the data by filling them with the mean values.
Splitting the data into features (X) and target (y). In this case, X will be a matrix of experience, test_score and interview_score, and y will be a vector of salary.
Creating and fitting a multiple linear regression model using sklearn.
Evaluating the model performance by calculating the coefficient of determination (R-squared) and the mean squared error (MSE).
Predicting the salary of new hires using the model.

### Files
-  `hiring.csv` The dataset used for training and testing the models. 
-  jupyter notebook `New_employee_salary_prediction.ipynb`

  
## How to Use
1. Clone this repository to your local machine.
2. Navigate to the respective project folder (e.g., `Housing_Price_Prediction` or `Car_Price_Prediction`).
3. Open the Jupyter Notebook files to explore the code, methodology, and results.
4. Feel free to modify the code, experiment with different techniques, and contribute to the projects.


# Project 3: Used Car Price Prediction

### Project Overview
This project is dedicated to predicting the prices of used cars based on various features contained within the cars.csv dataset. Encompassing the entire data science pipeline, this project spans from the initial data preprocessing stages to the final model evaluation, providing a comprehensive insight into used car price prediction.

## Objectives
-Data Cleaning and Preprocessing: Efficiently handle missing values, eliminate outliers, and remove unnecessary columns to ensure data quality.
-Exploratory Data Analysis (EDA): Visualize the dataset to unearth patterns, distributions, and relationships between features.
-Feature Engineering: Convert categorical variables through one-hot encoding and normalize numerical features.
-Model Selection and Training: Experiment with multiple regression models, train them, and evaluate their performance to select the optimal model.
-Model Evaluation: Assess the models using metrics like R^2, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to determine accuracy and reliability.
-Model Implementation: Serialize the best-performing model, making it ready for future predictions.

## Methodology
#### Data Cleaning and Preprocessing:
-Handled missing values using mode imputation.
-Identified and removed outliers based on the IQR method.
-Eliminated unnecessary columns, emphasizing the most impactful features

#### Exploratory Data Analysis (EDA):
-Visualized the distribution of car prices to understand their spread.
-Investigated the relationships between various features and the target variable, offering insights into potential correlations.

#### Feature Engineering:
-Transformed categorical variables using one-hot encoding.
-Scaled numerical attributes employing the StandardScaler, ensuring data standardization.

#### Model Selection and Training:
-Divided the data into distinct training and testing sets.
-Trained several regression models, including Random Forest Regressor, Linear Regression, Gradient Boosting Regressor, and XGBoost, comparing their performances.

#### Model Evaluation:
-Applied metrics such as R^2, MAE, and RMSE for thorough model evaluation.
-Identified XGBoost as the superior model due to its exceptional performance metrics

#### Model Implementation:
Serialized the finalized XGBoost model into `used_car_model.pkl` utilizing the pickle library for easy future deployments

### How to Use the Model

- Ensure the installation of necessary libraries: pandas, numpy, pickle, xgboost, and matplotlib.
-with open(`used_car_model.pkl`, 'rb') as file:
    ```loaded_model = pickle.load(file)```
- Prepare fresh data for prediction, ensuring it mirrors the format adopted during training.
- Implement one-hot encoding and scaling on the new data to match the training data structure.
Employ the loaded model to predict car prices

```predicted_price = loaded_model.predict(new_data_scaled)```

## Conclusion
This endeavour exemplifies the meticulous process involved in devising a machine learning model tailored for predicting used car prices. After rigorous optimization, the XGBoost model emerged as the top performer, making it an invaluable asset for platforms or applications offering used car price estimations. The serialized model ensures that this tool can be effortlessly integrated into various applications, promising accurate and reliable price predictions for used cars.



# Project 4: Glassdoor Data Science Jobs Analysis


### Project Overview:
This project offers an in-depth analysis of the data science job market, utilizing a dataset extracted from 1,500 job postings on Glassdoor.com. From initial data preprocessing to insightful visualizations, this project provides a comprehensive view of the data science job landscape.

### Objectives:
-Data Cleaning and Preprocessing: Efficiently handle missing values, address outliers, and ensure data quality for accurate analysis.
-Exploratory Data Analysis (EDA): Visualize the dataset to discover patterns, distributions, and relationships between features.
-Descriptive Analysis: Understand distributions like company ratings, and founding years, and delve into the top hiring companies and job locations.
-Feature Engineering: Focus on the most impactful features to drive insights and visualizations.
-Visualization: Craft informative visualizations that communicate the state and nuances of the data science job market.
-Insight Generation: Extract valuable insights from the visualizations and analysis, providing a snapshot of the current job market for data scientists.
### Methodology:
#### Data Cleaning and Preprocessing:
Managed missing values through median imputation and replaced placeholders.
Detected and capped outliers using the IQR method for numerical columns.
Streamlined the dataset by focusing on the most relevant features for analysis.

#### Exploratory Data Analysis (EDA):
Visualized the distribution of company ratings and founding years to understand their spread.
Investigated top hiring companies and job locations, offering insights into the demand landscape of the data science field.

#### Descriptive Analysis: 
Analyzed the top companies, job locations, and salary estimates to provide a comprehensive view of the data science job market.

#### Visualization:
Employed bar charts and histograms to effectively communicate the dataset's distributions and key patterns.

### How to Use the Analysis

-Ensure the installation of necessary libraries: pandas, numpy, matplotlib, and seaborn.
-There is CSV file for the project called `glassdoor_jobs.csv`
-Load the Jupyter notebook (`glassdoor_jobs_analysis.ipynb`) and run the cells to replicate the analysis.
-Customize or extend the analysis by adding your code or additional visualizations as needed.

### Conclusion
This project showcases a thorough analysis of the data science job market based on job postings from Glassdoor.com. Through meticulous preprocessing and insightful visualizations, a clear picture of the current state of data science jobs emerges. This analysis serves as a valuable resource for job seekers, hiring managers, and anyone interested in the data science field, offering a snapshot of the current trends, demands, and characteristics of the market.

## Contact
If you have any questions, or suggestions, or would like to collaborate, please feel free to contact me at shreeram.dangal@gmail.com.

Happy exploring and learning from these machine-learning projects!
