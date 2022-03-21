#Libraries
import numpy as np
import pandas as pd

#Data import/Data upload
df = pd.read_csv('datasets/first_datas.csv')

country = df.iloc[:,0:1].values
print(country)

from sklearn import preprocessing
#Label Encoding
labelEncoding = preprocessing.LabelEncoder()
country[:,0] = labelEncoding.fit_transform(df.iloc[:,0]) # Nominal datas is converted to number // 0 or 1
print(country)

"""
Example-LabelEncoder
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
"""
#OneHotEncoding

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)
"""
Example-OneHotEncoder
>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OneHotEncoder(handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names_out(['gender', 'group'])
array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)
"""