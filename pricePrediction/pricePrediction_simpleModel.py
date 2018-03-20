#importing required modules

import pandas as pd 
import matplotlib.pyplot as pyplot
import numpy as np


'''1.import the data set into the dataframes'''
#/Users/Lionel/Documents/Development/machineLearningConcepts/pricePrediction/

dc_listings = pd.read_csv('listings.csv')
# print dc_listings.shape
dc_listings.head()

'''first distance calculation'''
our_acc_value = 3
first_living_space_value = dc_listings.loc[0,'accommodates']

first_distance = np.abs(first_living_space_value - our_acc_value)
# print(first_distance)


dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)
dc_listings.distance.value_counts().sort_index()

dc_listings = dc_listings.sample(frac=1,random_state=0)
dc_listings = dc_listings.sort_values('distance')
dc_listings.price.head()

dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)
dc_listings['price']
mean_price = dc_listings.price.iloc[:5].mean()
mean_price



'''1. dropping the column 1 added during the model creation
2. data split -- training and test data '''

dc_listings.drop('distance',axis=1)

train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

print test_df.shape









