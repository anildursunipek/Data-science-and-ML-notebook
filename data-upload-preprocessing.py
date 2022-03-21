#Libraries
import numpy as np
import pandas as pd

#Data import/Data upload

df = pd.read_csv('datasets/first_datas.csv')

print(df[['height']])#Output column 'height'
print(df[['height','weight']])#Output column 'height' + 'weight'

#Missing Datas

df = pd.read_csv('datasets/missing_data.csv')
print(df)

#Alternative-1 for missing datas
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

age = df.iloc[:,1:4].values# 'Values' allows us to enclose all values in an array.
print(age)

imputer = imputer.fit(age[:,1:4])# We teach with fit
#Note: Fit ile nan olmayan tüm dataları toplayıp ortalamasını alıyoruz.
#--> We add the averages we get with fit to the places that are nan with Transform.
age[:,1:4] = imputer.transform(age[:,1:4])#We learn and implement with Transform
print(age)

#Alternativ-2 for missing datas (using pandas)
print(df.isnull())#Printing true np.nan values other false   df.isnotnull() reverse
print(df.isnull().sum())#Total null values

result = df.iloc[:,1:2].sum()
print(result)
"""
Output:
height    3276.0
weight    1151.0
age        558.0
"""
result = df.iloc[:,1:2].size
print(result)
result = df.iloc[:,1:2].isnull().sum()
print(result)


def means(df):
    calculate = df.iloc[:,1:2].sum()
    size = df.iloc[:,1:2].size - df.iloc[:,1:2].isnull().sum()
    return calculate/size
print(means(df))
print(df.fillna(value=(means(df))))
