# SALIFORT MOTORS PROJECT


The HR department has collected data from employees and the goals in this project is to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

 #### Variable                &    Description
 *satisfaction_level*     -   Employee-reported job satisfaction level [0–1].
 *last_evaluation*        -   Score of employee’s last performance review [0–1].
 *number_project*         -   Number of projects employee contributes to.
 *average_monthly_hours*  -   Average number of hours employee workedper month.
 *time_spend_company*     -   How long the employee has been with thecompany (years).
 *Work_accident*         -   Whether or not the employee experienced an accident while at work.
 *left*                   -   Whether or not the employee left the company.
 *promotion_last_5years*  -   Whether or not the employee was promoted in the last 5 years.
 *Department*             -   The employee’s department.
 *salary*                 - The employee’s salary (U.S. dollars.

 
 
 
 
 
 
 
 

 

### Import Packages


```python
!pip install xgboost

```

    Requirement already satisfied: xgboost in c:\users\user\anaconda3\lib\site-packages (2.1.2)
    Requirement already satisfied: numpy in c:\users\user\anaconda3\lib\site-packages (from xgboost) (1.26.4)
    Requirement already satisfied: scipy in c:\users\user\anaconda3\lib\site-packages (from xgboost) (1.11.4)
    


```python
# Import packages
# For data manipulation
import numpy as np
import pandas as pd

 # For data visualization
import matplotlib.pyplot as plt
import seaborn as sns
 # For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)
 # For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
 # For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
 f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree
 # For saving models
import pickle

import warnings
warnings.filterwarnings("ignore")



    
```


```python
#Load datase
df=pd.read_csv("HR_capstone_dataset.csv")
```

### Perform Exploratory Data Analysis (EDA) and data cleaning



```python
#show 1st 5 rows of the dataset
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     14999 non-null  float64
     1   last_evaluation        14999 non-null  float64
     2   number_project         14999 non-null  int64  
     3   average_montly_hours   14999 non-null  int64  
     4   time_spend_company     14999 non-null  int64  
     5   Work_accident          14999 non-null  int64  
     6   left                   14999 non-null  int64  
     7   promotion_last_5years  14999 non-null  int64  
     8   Department             14999 non-null  object 
     9   salary                 14999 non-null  object 
    dtypes: float64(2), int64(6), object(2)
    memory usage: 1.1+ MB
    

Descriptive Statistics about the dataset


```python
df.describe().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
      <td>14999.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.61</td>
      <td>0.72</td>
      <td>3.80</td>
      <td>201.05</td>
      <td>3.50</td>
      <td>0.14</td>
      <td>0.24</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.25</td>
      <td>0.17</td>
      <td>1.23</td>
      <td>49.94</td>
      <td>1.46</td>
      <td>0.35</td>
      <td>0.43</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.09</td>
      <td>0.36</td>
      <td>2.00</td>
      <td>96.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.44</td>
      <td>0.56</td>
      <td>3.00</td>
      <td>156.00</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.64</td>
      <td>0.72</td>
      <td>4.00</td>
      <td>200.00</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.82</td>
      <td>0.87</td>
      <td>5.00</td>
      <td>245.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>7.00</td>
      <td>310.00</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>






```python
#Inspecting column names
df.columns
```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
           'promotion_last_5years', 'Department', 'salary'],
          dtype='object')




```python
# Rename columns appropriately
df = df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
df.columns
```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_monthly_hours', 'tenure', 'work_accident', 'left',
           'promotion_last_5years', 'department', 'salary'],
          dtype='object')




```python
#Checking for null or missing values
df.isna().sum()
```




    satisfaction_level       0
    last_evaluation          0
    number_project           0
    average_monthly_hours    0
    tenure                   0
    work_accident            0
    left                     0
    promotion_last_5years    0
    department               0
    salary                   0
    dtype: int64



_there seems to be no null or missing values



```python
#checking for duplicate rows
df.duplicated().sum()
```




    3008



_3008 dulicate rows.


```python
df.shape
```




    (14999, 10)



_14999 rows and 10 columns in the dataset


```python
total_row=len(df)
```


```python
print(f'{total_row}')
```

    14999
    


```python
duplicate=df.duplicated().sum()
```


```python
percentage = (duplicate/total_row)*100
print(f'{duplicate} rows contains duplicate.That is {percentage:.2f}% of the data')
```

    3008 rows contains duplicate.That is 20.05% of the data
    


```python
#drop the duplicate and assign new result to a new df 
df1 = df.drop_duplicates(keep='first')
```


```python
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check if duplicate exist
df1.shape
```




    (11991, 10)



##### Checking for outliers


```python
# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(12,4))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()
```


    
![png](output_29_0.png)
    



```python
# Determine the number of rows containing outliers

# 25th percentile value in tenure
percentile25 = df1['tenure'].quantile(0.25)

# 75th percentile value in tenure
percentile75 = df1['tenure'].quantile(0.75)

# interquartile range in tenure
iqr = percentile75- percentile25

# Define the upper limit and lower limit for non-outlier values in tenure
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25- 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in tenure
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

# how many rows in the data contain outliers in tenure
print("Count of rows in the data containing outliers in tenure:", len(outliers))
```

    Lower limit: 1.5
    Upper limit: 5.5
    Count of rows in the data containing outliers in tenure: 824
    

#### Number of worker who stayed vs left


```python
#numbers of people who left vs. stayed
print(df1['left'].value_counts())
print()
print(df1['left'].value_counts(normalize=True).mul(100).round(2).astype('str')+'%')
```

    left
    0    10000
    1     1991
    Name: count, dtype: int64
    
    left
    0    83.4%
    1    16.6%
    Name: proportion, dtype: object
    


```python
 #Plot to visualise the average_monthly_hours, number_project against Left

 # Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (26,8))
#boxplot showing average_monthly_hours to  number_project`,
#comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project',hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='15')

# histogram showing distribution of number_project and comparing employees who stayed vs those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge',shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')
    
#Display the plots
plt.show()
```


    
![png](output_33_0.png)
    


It seems that there were two groups of workers who left the job, those who worked less than their peers with the same project and those who worked much more.


```python
# counts of stayed/left for employees with 6 projects
df1[df1['number_project']==6]['left'].value_counts()
```




    left
    0    455
    1    371
    Name: count, dtype: int64



Out of the employees who were assigned 6 projects, 455 stayed with the company, whereas 371 left.


```python
# counts of stayed/left for employees with 7 projects
df1[df1['number_project']==7]['left'].value_counts()
```




    left
    1    145
    Name: count, dtype: int64



All employees who were assign 7 project left.


```python
# scatterplot of average_monthly_hours versus satisfaction_level,against stayed vs left
plt.figure(figsize=(19, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level',hue='left', alpha=0.4)
#an assumption of 166.7hr/mo is used.
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Average monthly hours by satisfaction level', fontsize='14');
```


    
![png](output_39_0.png)
    


The scatter plot shows that large group of employees who worked for about 300 hrs were not satisfied with level below 0.2. Their working hours were roughly much more than their peers. some other workers were with in their working hours but was still not satisfied with a level around 0.4


```python
 # Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# boxplot showing distributions of satisfaction_level by tenure,against stayed vs left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left',orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5,ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')
plt.show();
```


    
![png](output_41_0.png)
    


Employees who departed can be grouped into two main categories: those with shorter tenures who were dissatisfied, and those with medium-length tenures who were very satisfied. Notably, employees who left at four years exhibited a particularly low level of satisfaction.while those who left after 5 years were highly statisfied.


```python
# mean and median satisfaction level of employees who left and those who stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>left</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.667365</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.440271</td>
      <td>0.41</td>
    </tr>
  </tbody>
</table>
</div>



As anticipated, employees who left have both mean and median satisfaction scores lower than those who stayed. Notably, for employees who stayed, the mean satisfaction score is somewhat below the median. This suggests that the distribution of satisfaction levels among those who remained might be left-skewed.


```python
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
 # Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]
 # Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1,hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5,ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people',fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1,hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4,ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people',fontsize='14');
```


    
![png](output_45_0.png)
    


The plots indicate that long-tenured employees were not predominantly among the higher-paid group


```python
 # Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation',hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');
```


    
![png](output_47_0.png)
    


The scatterplot reveals two distinct groups among the employees who left:Overworked employees who performed exceptionally well.
Employees who worked slightly less than the nominal average of 166.67 hours per month and had lower evaluation scores.
There appears to be a correlation between the number of hours worked and the evaluation score.
Notably, there is a low percentage of employees in the upper left quadrant of the plot, indicating that working long hours does not necessarily guarantee a high evaluation score.
The majority of employees in this company work well over 167 hours per month.


```python
#count on departments
df1["department"].value_counts()
```




    department
    sales          3239
    technical      2244
    support        1821
    IT              976
    RandD           694
    product_mng     686
    marketing       673
    accounting      621
    hr              601
    management      436
    Name: count, dtype: int64




```python
 # Create stacked histogram to compare department distribution of employees who left to that of employees who didn't
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1, hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='horizontal')
plt.title('Counts of stayed/left by department', fontsize=14);
```


    
![png](output_50_0.png)
    


There doesn’t seem to be any department that differs significantly in its proportion of employees who left to those who stayed.


```python
#plt.figure(figsize=(16, 9))
#this isolate only float and int columns
df1_n = df1.select_dtypes(include=[float,int])

heatmap = sns.heatmap(df1_n.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
```


    
![png](output_52_0.png)
    


From the heatmap, last evalution,number of project and average monthly hours have high positive correlation and left has a negative correlation with the satisfaction level.

### Model Building

since the variable to predict is a categorical variable (left) ,Logistic regression model or Tree based ML model should be used.

Building a Logistic regression model


```python
 #Copy the dataframe
df_enc = df1.copy()
 # Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
 df_enc['salary'].astype('category')
 .cat.set_categories(['low', 'medium', 'high'])
 .cat.codes
 )
 # Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)


# Convert only the dummy columns to int (those columns created by get_dummies)
df_enc[df_enc.columns[df_enc.columns.str.contains('department')]] = df_enc[df_enc.columns[df_enc.columns.str.contains('department')]].astype(int)


 # Display the  new dataframe
df_enc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project','average_monthly_hours', 'tenure']].corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()
```


    
![png](output_58_0.png)
    



```python
# Select rows without outliers in `tenure` and save resulting dataframe in a␣new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <=upper_limit)]
 
# Display first few rows of new dataframe
df_logreg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Isolate the outcome variable
y = df_logreg['left']
 # Display first few rows of the outcome variable
y.head()
```




    0    1
    2    1
    3    1
    4    1
    5    1
    Name: left, dtype: int64




```python
# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)
 # Display the first few rows of the selected features
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y, random_state=42)
```


```python
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train,y_train)
```


```python
# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)
```


```python
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
 # Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,display_labels=log_clf.classes_)
 # Plot confusion matrix
log_disp.plot(values_format='')
 # Display plot
plt.show()
```


    
![png](output_65_0.png)
    


2165: The number of people who did not leave that the model accurately predicted did
 not leave.
156: The number of people who did not leave the model inaccurately predicted as
 leaving.
 348: The number of people who left that the model inaccurately predicted did not leave
 123: The number of people who left the model accurately predicted as leaving


```python
df_logreg['left'].value_counts(normalize=True).mul(100).round()
```




    left
    0    83.0
    1    17.0
    Name: proportion, dtype: float64



 Approximately 83%-17% split. 


```python
# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))
```

                               precision    recall  f1-score   support
    
    Predicted would not leave       0.86      0.93      0.90      2321
        Predicted would leave       0.45      0.27      0.34       471
    
                     accuracy                           0.82      2792
                    macro avg       0.66      0.60      0.62      2792
                 weighted avg       0.79      0.82      0.80      2792
    
    

The classification report above shows that the logistic regression model achieved a precision of 79%,
 recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%,on the text set. However, if it’s most
 important to predict employees who leave, then the scores are significantly lower. The model predict would not leave is high score.

###  Conclusion, Recommendations.
 The models and the feature importances extracted from the models confirm that employees at the
 company are overworked.
 To retain employees, the following recommendations could be suggusted:
1. Cap the number of projects that employees can work on.
2. Consider promoting employees who have been with the company for atleast four years, or
 conduct further investigation about why four-year tenured employees are so dissatisfied.
3. Either reward employees for working longer hours, or don’t require them to do so.
4. If employees aren’t familiar with the company’s overtime pay policies, inform them about
 this. If the expectations around workload and time off aren’t explicit, make them clear.
4. Hold company-wide and within-team discussions to understand and address the company
 work culture, across the board and in specific contexts.
 
5. Highevaluation scores should not be reserved for employees who work 200+ hours per month.Consider a proportionate scale for rewarding employees who contribute more/put in more effort


```python

```
