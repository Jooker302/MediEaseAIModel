import pandas as pd
import numpy as np

def get_unique_values(dataframe,col_list):
    for col in col_list:
        print('Column:',col)
        print('Unique values:',dataframe[col].unique())
        print('\n')

def get_value_counts(dataframe,col_list):
    for col in col_list:
        print(dataframe[col].value_counts())
        print('\n')



# diabetes = pd.read_csv('/kaggle/input/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
diabetes = pd.read_csv('diabetes_prediction_dataset.csv')
# diabetes.head()

diabetes.info()

diab_num = diabetes[['age','bmi','HbA1c_level','blood_glucose_level']]

diab_num.describe()

# get_unique_values(diabetes,['gender','hypertension','heart_disease','smoking_history','diabetes'])

get_value_counts(diabetes,['gender','hypertension','heart_disease','smoking_history','diabetes'])


# diab_num.info()