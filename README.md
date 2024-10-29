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
