import pandas as pd
from sklearn.linear_model import LogisticRegression

# Create df
train = pd.read_csv('data/titanic.txt')

# Drop null values
train.dropna(inplace = True)

# features and target
target = 'Survived'
features = ['Pclass', 'Age','SibSp', 'Fare']

# X matrix, y vector
X = train[features]
y = train[target]

# model
model = LogisticRegression()
model.fit(X,y)
model.score(X,y)

# Saving the model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
