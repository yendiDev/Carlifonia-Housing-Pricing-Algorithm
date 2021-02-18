"""
Main program to utilise the machine learning model created in model.py file
by Clinton Anani
email: kceequan01@gmail.com
github: github.com/yendiDev
AI is coming!

"""

import pickle
import numpy as np

# import machine learning model
pickle_in = open('prices_model.pickle', 'rb')
model = pickle.load(pickle_in)

# program intro
print('#############################################################################')
print('WELCOME TO THE CALIFORNIA HOUSING PRICES PREDICTION PROGRAM')
print('This model has an accuracy of about 45-50%, so values will be obscured')
print('#############################################################################\n\n')

# receive user median income

income = float(input('What is your median income?: '))
# parse income into the right format
income = np.array(income).reshape(-1, 1)

# make prediction
prediction = model.predict(income)
price = prediction[0].round(2)

# print out results
print('Based on your median income, this model predicts you\'ll be paying an approximated amount of ${0} '
      'for your house in Carlifonia'.format(price))