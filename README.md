# cse151a-group-project
CSE 151A group project

# How will you preprocess your data?

First, we will need to replace null values for the feature Mental_Health_Condition and Physical_Activity with actual values.

For the null values in Mental_Health_Condition, we will label null values as "Healthy" as a null value here indicates healthy mental health.

For the null values in Physical_Activity, we will label null values as "No Exercise" as a null value here indicates no physical activity.

We won’t be scaling our dataset because the number of observations for each category in “Mental_Health_Condition” is evenly distributed. We believe the dataset is already balanced in terms of the output labels.

For the non numerical ordinal columns (stress_level, sleep_quality, and physical_activity), we will encode them in the range of 1 to 3.

We will drop the employee_id column as it is irrelevant to the target column “Mental_Health_Condition”.

For the nominal categories that are not binary (Gender, Job_Role, Industry, Work_Location, Productivity_Change, Region), we will be doing one-hot encoding.

Jupyter notebook google colab link: https://colab.research.google.com/drive/1iie7QJZn2lgUzOD9kKXeIcS07fiX64wC?usp=sharing
Jupyter notebook file locally is called milestone2.ipynb

# MILESTONE 3 ADDITIONS BELOW

One additional preprocessing step that we added was consolidating people with mental health conditions into one group and "healthy" people in one group to simplify our classification task.

### Steps 1, 2, and 3 are on the jupyter notebook local file milestone3.ipynb

# 3: Evaluate your model and compare training vs. test error
We found our test accuracy to be 0.766, while our training accuracy was 0.762, slightly better but about the same. A full classification report was printed out for both training and test metrics in the ipynb notebook.

#4: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?
Our model fits on the very left side, as evident by the fact that our test and training accuracy are essentially equal. This indicates our model is underfitting. We are thinking of tying out a logistic regression and naive bayes because our recall is 0

# 5: 
Our jupyter notebook is on this repo locally as milestone3.ipynb but we also have a Google colab set up for it: https://colab.research.google.com/drive/16FCSKdOQAi7UqrjR4zN4H9yZvxkhOWjW?usp=sharing

# 6 Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?
The conclusion of our 1st model is that it seems to always be predicting "unhealthy." This might be due to highly overlapping features between the two classes. One possible solution to this phenomenon is to increase the C (regularization) value to penalize the model when it guesses wrong. We hope that this encourages the model to predict healthy in more cases. We want to try oversampling as well since our data set is quite unbalanced, and SVMs do not like unbalanced data.

__Note:__
We find that accuracy metrics point toward the fact that our model simply predicts all observations as "unhealthy". For example, when we run our classification report on a binary target value (healthy or unhealthy), precision and recall are 0 for predicting healthy and (.750, 1.0) for predicting unhealthy. Our accuracy is also 0.75 for our test data. We are fully convinced that our data set was created artificially, and contains no actual correlation between any features. We expressed this concern in a piazza post during our milestone 2 submission, and responses from instructors gave us the go ahead to keep the data set. The TA told us to drop features we were uncomfortable with but it appears that all the features are random.
