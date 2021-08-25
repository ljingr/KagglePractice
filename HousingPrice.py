#%%
import pandas as pd
from scipy.sparse.construct import random 
from sklearn.model_selection import train_test_split

# %%
# read the data
X_train = pd.read_csv('train.csv',index_col='Id')
X_test = pd.read_csv('test.csv',index_col='Id')
# %%
X_train.head()
# %%
# obtain target and predictors
y_train = X_train.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_select = X_train[features]
X_test_select = X_test[features]
# %%
# break off validation data
X_train_true, X_valid, y_train_true, y_valid = train_test_split(X_select,y_train,train_size = 0.8, test_size = 0.2, random_state = 0)
# %%
X_train_true.head()
# %%
from sklearn.ensemble import RandomForestRegressor
# define the model
model_1 = RandomForestRegressor(n_estimators=50,random_state = 0)
model_2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
model_3 = RandomForestRegressor(n_estimators = 100, criterion = 'mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators = 200, min_samples_split = 20, random_state = 0)
model_5 = RandomForestRegressor(n_estimators = 200, max_depth = 7, random_state = 0)
models = [model_1, model_2, model_3, model_4, model_5]

# %%
from sklearn.metrics import mean_absolute_error

def score_model(model, X=X_train_true, y = y_train_true, X_val = X_valid, y_val = y_valid):
    model.fit(X,y)
    pred = model.predict(X_val)
    return mean_absolute_error(pred,y_val)


# %%
import numpy as np
mae = np.zeros(len(models))
for i in range(len(models)):
    mae[i] = score_model(models[i])
    print('mae of model {} is {}'.format(i+1,mae[i]))
# %%
best_model = models[np.argmin(mae)]

# %%
best_model.fit(X_select,y_train)
pred_test = best_model.predict(X_test_select)

# %%
output = pd.DataFrame({"Id":X_test_select.index,"SalePrice": pred_test})
output.to_csv('test_result.csv',index = False)

# %%
