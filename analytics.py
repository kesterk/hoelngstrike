import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()
USAhousing.describe()
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#creating_&_training_model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(X_train,y_train)
