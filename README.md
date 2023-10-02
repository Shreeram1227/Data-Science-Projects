
## Project 1: Per capita income prediction

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
- `Linear_Regression_per capita income-Copy1`: Jupyter Notebook containing the linear regression implementation.
- `multiple_linear_regression.ipynb`: Notebook for the multiple linear regression analysis.
- `gradient_descent.ipynb`: Notebook demonstrating the use of gradient descent.

  ## Project 2: Predicting Salary of New Hires by using Multiple Linear Regression

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
-  hiring.csv The dataset used for training and testing the models.
-  jupyter notebook multiple_linear_regression.ipynb

  



## How to Use
1. Clone this repository to your local machine.
2. Navigate to the respective project folder (e.g., `Housing_Price_Prediction` or `Car_Price_Prediction`).
3. Open the Jupyter Notebook files to explore the code, methodology, and results.
4. Feel free to modify the code, experiment with different techniques, and contribute to the projects.

## Contact
If you have any questions, suggestions, or would like to collaborate, please feel free to contact me at shreeram.dangal@gmail.com.

Happy exploring and learning from these machine learning projects!
