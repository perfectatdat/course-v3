# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:34:42 2019

@author: GhoshalK
"""

import pandas as pd
import numpy as np

housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
housing = pd.read_csv("C:\\Users\\GhoshalK\\Documents\\Python\\housing.csv")

###Import files from url##
import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()
    
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
##################
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
housing["latitude"].describe()
housing["total_rooms"].describe()
housing["median_income"].describe()

import matplotlib.pyplot as plt
housing.hist(figsize = (10,10))


####Median Income used for stratified sampling####
###The range of the median income is reduced###
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].value_counts()/len(housing)
####Using where clause clipping outlier values###reducing the categories to 5##
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)
#housing["income_cat"].where(housing["income_cat"]>5,5.0,False)
housing['income_cat'].value_counts()

####Doing random sampling#####
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size = 0.2, random_state = 100)

len(train_set)
len(test_set)

housing["income_cat"].value_counts()/len(housing)
train_set["income_cat"].value_counts()/len(train_set)
test_set["income_cat"].value_counts()/len(test_set)

###Doing stratified sampling##
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=100)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts()/len(housing)
strat_train_set["income_cat"].value_counts()/len(strat_train_set)
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

###Dropping the income category column###
strat_train_set.info()
strat_train_set.drop(["income_cat"], axis=1, inplace=True)
strat_test_set.drop(["income_cat"], axis=1, inplace=True)

####creating a copy of the train set ###
housing = strat_train_set.copy()
###scatter plot####
housing.plot(kind = "scatter", x= "longitude", y = "latitude")
###Identifying the point with higher number of observations###
housing.plot(kind = "scatter", x= "longitude", y = "latitude", alpha=0.1) 

housing.plot(kind = "scatter", x= "longitude", y = "latitude", alpha=0.4, s = housing["population"]/100,label ="Population", c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar =True, figsize = (10,10))
plt.legend()

#####Finding important variables#####
#####finding correlation matrix###
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#####Creating new variables####
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

####Separating the target variable###
housing = strat_train_set.drop(["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_labels.describe()

# removing rows which have missing values in total bedrooms column#
housing.dropna(subset=["total_bedrooms"]) 
##removing the entire total bedrooms column## 
housing.drop("total_bedrooms", axis=1) # option 2
####Filling missing values with median
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

###using the imputer class for imputation##
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")
###Applying median imputation only on the numeric variables##
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

imputer.statistics_ ###The median values for all numeric cols### 
housing_num.median().values
####Imputing the missing values with median##
X = imputer.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns)
housing_tr["total_bedrooms"].describe()
housing["total_bedrooms"].describe()

#####Converting categorical variables to numeric###
####Using label encoder###
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
encoder.classes_

#####using one-hot encoder####
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
###Sparse one hot encoder###
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
###converting to numpy array##
housing_cat_1hot.toarray()
##############################

#####Doing the one hot encoding in one shot####
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer(sparse_output =True) ##sparse_output = True for sparse matrix
housing_cat_1hot = encoder.fit_transform(housing_cat)


######
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer',Imputer(strategy = "median")),
        ('std_scalar',StandardScaler())
        ])

housing_num_scaled = pd.DataFrame(num_pipeline.fit_transform(housing_num))

housing_num.describe()
housing_num_scaled.describe()

from sklearn.base import BaseEstimator, TransformerMixin

###Subsets the data based on column type###
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

####Writing a custom class to convert from category to binary ####
class CustomLabelEncode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return OneHotEncoder().fit_transform(LabelEncoder().fit_transform(X).reshape(-1,1));
    
num_attributes = list(housing_num)
cat_attributes = ["ocean_proximity"]

cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attributes)),
#        ('label_binarizer',LabelBinarizer(sparse_output=True)),
        ('one_hot_encoder',CustomLabelEncode()),
        ])

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attributes)),
        ('imputer',Imputer(strategy = "median")),
        ('std_scalar',StandardScaler()),
        ])

full_pipeline = FeatureUnion(transformer_list = [
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)
        ])

cat_pipeline.fit_transform(housing)
num_pipeline.fit_transform(housing)

####Nested functions###
LabelBinarizer(sparse_output =True).fit_transform(DataFrameSelector(cat_attributes).fit_transform(housing))

####Running the full pipeline#########
housing_prepared = full_pipeline.fit_transform(housing)

#LabelBinarizer().fit_transform(housing["ocean_proximity"])

#OneHotEncoder().fit_transform(LabelEncoder().fit_transform(housing_cat).reshape(-1,1))

####Linear regression####
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg,housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)

##############################

####Decision tree regressor #######
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation", scores.std())
    
display_scores(rmse_scores)
#######################################

#####RandomForest Regressor############
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)

####Grid search for hyperparameter tuning######
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'n_estimators':[30,100,500], 'max_features':[2,4,6,8]},
        {'bootstrap':[False], 'n_estimators':[30,100],'max_features':[2,3,4]}
        ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid, cv = 5, scoring = 'neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)
###Tuned hyperparameters###
grid_search.best_params_
####best model#####
grid_search.best_estimator_
    
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

feature_importance =grid_search.best_estimator_.feature_importances_

list(housing_prepared.columns.values)
pd.DataFrame(housing_prepared.todense()).columns
type(housing)
housing.columns

sorted(zip(feature_importance, list(housing.columns)), reverse=True)

final_model = grid_search.best_estimator_


#####Applying on the test set##########
X_test = strat_test_set.drop("median_house_value", axis =1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)

########################################

from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(housing_prepared,housing_labels)

####Grid search for hyperparameter tuning######
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel':["linear", "poly", "rbf", "sigmoid"], 'C':[1.0,10.0,100.0], 'epsilon': [0.1,0.01,0.001]},
#        {'shrinking':[False],'kernel':["linear", "poly", "rbf", "sigmoid", "precomputed"], 'C':[1.0,10.0,100.0],'epsilon':[0.1,0.01,0.001]}
        ]
svr_reg = SVR()
grid_search = GridSearchCV(svr_reg,param_grid, cv = 5, scoring = 'neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

#####
import parfit.parfit as pf
import numpy as np
from sklearn.model_selection import ParameterGrid
# Necessary if you wish to run each step sequentially
from parfit.fit import *
from parfit.score import *
from parfit.plot import *
from parfit.crossval import *

from sklearn.metrics import *

####Creating validation sets for hyperparameter tuning###
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(strat_train_set,test_size = 0.2, random_state = 100)

X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_val = val_set.drop('median_house_value', axis=1)
y_val = val_set['median_house_value']

X_train= full_pipeline.fit_transform(X_train)
X_val= full_pipeline.fit_transform(X_val)

param_grid = ParameterGrid({
        'kernel':["linear"],
        'C':[99000,100000,110000],
        'epsilon': [0.1]
        })

svr_reg = SVR()
best_model, best_score, all_models, all_scores = pf.bestFit(svr_reg, param_grid,X_train, y_train, X_val,y_val,metric=mean_squared_error, greater_is_better=False, scoreLabel='mse')

print(best_model)
svr_rmse = np.sqrt(best_score)

param_grid ={
        'n_estimators':[30,100,500],
        'max_features':[2,4,6,8]
         }
#        {'bootstrap':[False], 'n_estimators':[30,100],'max_features':[2,3,4]}

param_grid = ParameterGrid(param_grid)

pf.crossvalModels(forest_reg,housing_prepared,housing_labels, param_grid, nfolds=5, metric=mean_squared_error,n_jobs=-1, verbose=0)

from sklearn.model_selection import ParameterGrid

forest_reg = RandomForestRegressor()

pf.bestFit(forest_reg, param_grid , housing_num, housing_labels, nfolds=2, predict_proba = False, metric = mean_squared_error, greater_is_better = False, verbose=1)



#####To save the session###
import dill                            #pip install dill --user
filename = 'C:\\Users\\GhoshalK\\Documents\\Python\\housingData.pkl'
dill.dump_session(filename)

# and to load the session again:
dill.load_session(filename)

import pickle
with open(filename, 'rb') as f:
    var_you_want_to_load_into = pickle.load(f)
