#importing required modules

import pandas as pd 
import matplotlib.pyplot as pyplot
import numpy as np


'''1.import the data set into the dataframes'''

dc_listings = pd.read_csv('/Users/Lionel/Documents/Development/machineLearningConcepts/pricePrediction/listings.csv')
print dc_listings.shape
dc_listings.head()



our_acc_value = 3
first_living_space_value = dc_listings.loc[0,'accommodates']

first_distance = np.abs(first_living_space_value - our_acc_value)
print(first_distance)







