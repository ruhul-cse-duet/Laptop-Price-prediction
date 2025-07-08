
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor,VotingRegressor,StackingRegressor

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


## Model training.....................Function.............................
#"Linear_Regression":LinearRegression(),

def linear_reg(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),[0,1,3,4,5,9,10,11]),
    ], remainder='passthrough')

    step_2 = LinearRegression(
        fit_intercept=True,
        copy_X=True,
        n_jobs=-1,
        positive=False
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train,y_train)
    #pipe.score(X_test,y_test)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))

#.............////////////....................................................................//////////
def Lasso_reg(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = Lasso(
        alpha=1.0,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=500,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=42,
        selection="cyclic",
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))

#................................................./////////////////////...........................................
def Ridge_reg(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = Ridge(
        alpha=10,
        fit_intercept=True,
        copy_X=True,
        max_iter=500,
        tol=1e-4,
        positive=False,
        random_state=42,
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def ElasticNet_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = ElasticNet(
        alpha=5.0,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=700,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=42,
        selection="cyclic",
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def DecisionTreeRegressor_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')
    step_2 = DecisionTreeRegressor(
        criterion='squared_error',
        splitter='best',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))



def AdaBoostRegressor_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')
    step_2 = AdaBoostRegressor()

    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def GradientBoost_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = GradientBoostingRegressor()

    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def XGBRegressor_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = XGBRegressor(
        max_depth=10,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.001,
        gamma=0.01,
        max_delta_step=2,
        eval_metric="rmse",
        enable_categorical=True,
        device='cuda'
    )

    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def LGBMRegressor_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def CatBoostRegressor_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = CatBoostRegressor(
        iterations=3500,
        depth=12,
        loss_function='RMSE',
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='RMSE',
        silent=True
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))


def RandomForest_model(X_train,X_test,y_train,y_test):
    step_1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         [0, 1, 3, 4, 5, 9, 10, 11]),
    ], remainder='passthrough')

    step_2 = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([
        ('step_1', step_1),
        ('step_2', step_2),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R² Score:", r2_score(y_test, y_pred))

