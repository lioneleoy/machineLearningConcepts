#importing required modules

import pandas as pd 
import matplotlib.pyplot as pyplot
import numpy as np

'''1.import the data set into the dataframes'''
#/Users/Lionel/Documents/Development/machineLearningConcepts/pricePrediction/

dc_listings = pd.read_csv('listings.csv')
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)
# print dc_listings.shape
dc_listings.head()

'''first distance calculation'''
our_acc_value = 3
first_living_space_value = dc_listings.loc[0,'accommodates']

'''1. dropping the column 1 added during the model creation
2. data split -- training and test data '''
# dc_listings.drop('distance',axis=1)
train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

'''creating function of the simple model, refering the pricePrediction_simpleModel'''
def predictPrice(new_listing_value, feature_columns):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_columns] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return predicted_price


'''predictin using the simple model'''
test_df['predicted_price'] = test_df.accommodates.apply(predictPrice, feature_columns = 'accommodates')
# print test_df['predicted_price'].head()
'''calculating RMSE -- Root Mean Square Error
1. difference between actual and predicted value
2. squaring the distance
3. taking mean of all the squared values
4. taking squared root of the mean'''

test_df['squared_error'] = (test_df['predicted_price'] - test_df['price']) ** (2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2.0)


'''comparing different models RMSE'''

for feature in ['accommodates','bedrooms','bathrooms','number_of_reviews']:
    test_df['predicted_price'] = test_df.accommodates.apply(predictPrice, feature_columns=feature)
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price']) ** (2)
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1/2.0)
    print("RMSE for the {} column: {}".format(feature,rmse))
