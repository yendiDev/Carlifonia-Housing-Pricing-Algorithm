import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

# import data set
housing = pd.read_csv('housing copy.csv')

# predict
predict = 'median_house_value'

# create model
model = linear_model.LinearRegression()

largest = 0
# run continuous iterations to find the largest accurate value
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(housing.median_income, housing.median_house_value, test_size=0.1)

    # fit data into model
    model.fit(np.array(x_train).reshape(-1, 1), y_train)

    acc = model.score(np.array(x_test).reshape(-1, 1), y_test)
    if(acc > largest):
        largest = acc

# save model after iterations
with open('prices_model.pickle', 'wb') as f:
    pickle.dump(model, f)



