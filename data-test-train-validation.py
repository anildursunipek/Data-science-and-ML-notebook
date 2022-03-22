import pandas as pd
from sklearn import preprocessing

#Data import/Data upload
df = pd.read_csv('datasets/first_datas.csv')
country = df.iloc[:,0:1].values
#Label Encoding
labelEncoding = preprocessing.LabelEncoder()
country[:,0] = labelEncoding.fit_transform(df.iloc[:,0]) # Nominal datas is converted to number // 0 or 1
#OneHotEncoding
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
# DataFrame Combination
result = pd.DataFrame(data=country, index=range(22),columns=['fr','tr','us'],dtype=('int'))
result2 = df.drop(columns=('country'))
concat = pd.concat([result,result2],axis=1)
#Data train-test split
from sklearn.model_selection import train_test_split

independent_veriables = concat.drop('gender',axis=1)
dependent_veriables = concat['gender']

x_train, x_test, y_train, y_test = train_test_split(independent_veriables,dependent_veriables,test_size=0.33,random_state=0)

#The first parameter consists of independent variables.
#The second parameter consists of dependent variables.
#Test size = What ratio will be between test and train data
#Random_state = data split from a random place

#Feature scaling - Standardization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)