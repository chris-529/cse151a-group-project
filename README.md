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

# Milestone 3 additions based on Gradescope feedback:

As per the feedback, we decided to check model results using a linear kernal and trying out different C regularization values. We picked the model with the highest accuracy, which was using a linear kernel and C value of 1.0. We also oversampled our data using SMOTE. As per the feedback, we also performed coefficient analysis on this model we selected by printing out every coefficient along with the feature name. A list of all coefficients can be found in the notebook under the header "Milestone 3 using linear kernal and finding the optimal regularization term (based on Gradescope feedback). Also oversampling and coefficient analysis."

We also looked at our data's correlation matrix through a seaborn heatmap, as seen here:
![](heatmap.png)

According to the heatmap above, most of the columns are not correlated at all. The squares of heat in our heatmap are the one-hot-encoded columns that correspond to the same feature. The heatmap contains all of the same heat with some noise, furthering our belief that our dataset was generated synthetically and randomly.

Our milestone4.ipynb notebook also tries to replicate the process that the original dataset author might have used. We find that if we randomly generate a dataset, and assign features to observations uniformly at random, we obtain a dataset similar to the one that we have been investigating. Namely, we get a very similar number of contradictory observations, and a similar distribution of all feature value occurrences.

milestone4.ipynb google colab link for easy access: https://colab.research.google.com/drive/1auR0nPVbMbBUuum9VPBjQTmcdvwHZDLW?usp=sharing

# Milestone 4 :

1.

We decided for our 2nd model to use a random forest classifier.

We ran GridSearchCV from sklearn to learn the optimal parameters to use in our model to avoid overfitting. This method yielded the following parameters:
Best Parameters:
{'max_features': None,
'min_samples_leaf': 15,
'n_estimators': 100}

However, we found that setting n_estimators = 10 instead of 100 yields better model performance, as indicated by the fact that true negatives now appear. Perhaps due to the fact that a higher n_estimators value can lead to overfitting.

2.

Our model yielded a training accuracy of 0.7710, while our test accuracy 0.7667. Additionally, our model now yields a non-zero recall for the negative classes, indicating that our model is not naively predicting true each time. Our recall on our model for the positive class is very high, sitting at 0.999, meaning it successfully predicts almost all instances of positive classes as positive. We also performed coefficient analysis on our model, and found that the top 5 features were Age, Hours_Worked_Per_Week, Years_of_Experience, Number_of_Virtual_Meetings, and Social_Isolation_Rating.

3.

Our model lies in the "ideal range" of the fitting graph because we do not have substantial overfitting, as evidenced by our training and testing accuracies differing. Furthermore, if we were to increase the complexity of our dataset, we find that we experience overfitting (n_estimates=100 causes overfitting).

4.

Our code for milestone 4 is in the local file milestone4.ipynb, but it is also available on Google colab through the link https://colab.research.google.com/drive/1auR0nPVbMbBUuum9VPBjQTmcdvwHZDLW?usp=sharing.

5.

We conclude that this second model, using a random forest classifier, performs better than our previous SVM model and performs better than random. We are now able to successfully classify some negatives, as opposed to just predicting positive for everything like our previous model did. We also get a higher accuracy on the test set with our second model compared to our first model. We could further improve our model by balancing the data more, as well as dropping features to reduce noise.

Model 1 methods:

Using a SVM tries to create a decision boundary of our data, however this did not work well for us since there is a lot of overlap in our data.

Model 2 methods:

Using a random forest classifier, similar to a decision tree in class, gave us better results since it does not depend on a decision boundary and therefore works better on highly overlapping data.

6.

We have printed an example of a true positive, true negative, false positive, and false negative at the bottom of our milestone4.ipynb file.
