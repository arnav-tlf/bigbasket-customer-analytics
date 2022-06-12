# To do's 
# 1) Create a class for forecasting score that allows us to customize buckets 
# 1a) Add the current metrics in there as static functions 
# 2) Add the typing module to HDI 
# 3) No provision for time series CV - just a simple train val test split. 
# 4) Currently, HDI python's has some weird issue when we define a function to get an arbitary number of intiial inputs def _f(*args,... ). Refactor are shift.
# 5) Serialize the pipelien and save it so we can call it up for prediction


import numpy as np
from numpy.core import numeric
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, PoissonRegressor, GammaRegressor
import xgboost as xgb
import types
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Validate if column is datetime 
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def forecast_buckets(x):
    if x < 0.2:
        return 2
    elif x < 0.5:
        return 1
    elif x < 0.8:
        return 0
    else:
        return -1

def forecasting_score(actuals, preds,score = True):
    """We divide the scores according to the buckets defined in forecast_buckets"""
    pcent = np.abs(preds/actuals - 1)
    pcent_buckets = pcent.apply(forecast_buckets).value_counts()/pcent.shape[0]
    
    if score:
        _score = 0
        for _s, _w in zip([2,1,0,-1],(2,1,-1,-1.5)):
            if _s in pcent_buckets:
                _score += _w*pcent_buckets[_s]
        # scaling between 0-1
        _score = (_score + 1.5)/(3)
        return _score
    else:
        return pcent_buckets

def error_metrics(actuals, preds, metrics  = ['forecasting_score', 'mae']):
    """Get a list of various forecasting error metrics"""
    results = []
    names = []
    for _m in metrics:
        names.append(_m.__name__ if isinstance(_m, types.FunctionType) else _m)
        if isinstance(_m, types.FunctionType):
            results.append(_m(actuals,preds))
        elif _m == 'forecasting_score':
            results.append(forecasting_score(actuals,preds))
        elif _m == 'forecasting_report':
            results.append(forecasting_score(actuals,preds,False))
        elif _m == 'mae':
            results.append(mean_absolute_error(actuals,preds))
        elif _m == 'mse':
            results.append(mean_squared_error(actuals,preds))
        elif _m == 'rmse':
            results.append(mean_squared_error(actuals,preds, squared=False))
        elif _m == 'r2':
            results.append(r2_score(actuals,preds))
    return {key:val for key, val in zip(names,results)}


def forecasting_grid_search(train,
                            val,
                            hyperparams,
                            y_var,
                            numeric_features = [],
                            categorical_features = [],
                            weights = None,
                            metrics = ['forecasting_score','mae','r2','rmse']):

    # List to concat results over all parameters 
    results = []

    # If no weights are passed, set all obs to equal weight 
    if weights is None:
        weights = np.ones(train.shape[0])

    for idx, curr_hyperparams in enumerate(hyperparams):
        if idx % 10 == 0:
            print('Param {0} of {1}'.format(idx + 1, len(hyperparams)))
        
        # General transformations 
        model_type = curr_hyperparams['type']
        _transform = curr_hyperparams['transformations']
        missing_val_impute = curr_hyperparams['missing_values']
        model_params = {key:curr_hyperparams[key] for key in curr_hyperparams.keys() if key not in ['type','transformations','missing_values']}

        X_train, X_val = fit_and_transform(train,val,transformation = _transform,numeric_features = numeric_features, categorical_features = categorical_features)
        X_train, X_val = impute_missing(X_train, X_val, missing_val_transformation=missing_val_impute,numeric_features=numeric_features, categorical_features=categorical_features)

        if model_type == 'xgboost':
            
            # Model specific params 
            num_boost_rounds = model_params['num_boost_round']
            model_params = {key:model_params[key] for key in model_params.keys() if key not in ['num_boost_round']}

            dmat_train = xgb.DMatrix(X_train, label = train[y_var], feature_names = X_train.columns, weight = weights)
            dmat_val = xgb.DMatrix(X_val, label = val[y_var], feature_names=X_train.columns)
            booster = xgb.train(model_params, dtrain = dmat_train,
                                evals = [(dmat_train, 'train'), (dmat_val, 'val')],
                                num_boost_round=num_boost_rounds,
                                early_stopping_rounds=20,
                                verbose_eval = False)

            curr_preds = booster.predict(dmat_val)
            curr_hyperparams.update({'num_boost_round':booster.best_ntree_limit})
            results.append((error_metrics( val[y_var],curr_preds,metrics), curr_hyperparams))
            
        elif model_type =='xgb_sklearn':
            cv_reg = xgb.XGBRegressor(random_state = 42,**model_params)
            cv_reg.fit(X_train, train[y_var],
                    eval_set=[(X_train, train[y_var]), (X_val, val[y_var])],
                    early_stopping_rounds=50,
                    verbose=False)
            curr_preds = cv_reg.predict(X_val)
            curr_hyperparams.update({'n_estimators':cv_reg.best_ntree_limit})
            results.append((error_metrics( val[y_var],curr_preds,metrics), curr_hyperparams))
        
        elif model_type == 'elasticnet':
            elr = ElasticNet(random_state = 42,**model_params)
            elr.fit(X_train, train[y_var],sample_weight=weights)
            results.append((error_metrics( val[y_var],elr.predict(X_val),metrics), curr_hyperparams ))
            
        elif model_type == 'poisson':
            plr = PoissonRegressor(**model_params)
            plr.fit(X_train, train[y_var],sample_weight=weights)
            results.append((error_metrics( val[y_var],plr.predict(X_val),metrics), curr_hyperparams ))

        elif model_type == 'gamma':
            glr = GammaRegressor(**model_params)
            glr.fit(X_train, train[y_var],sample_weight=weights)
            results.append((error_metrics( val[y_var],glr.predict(X_val),metrics), curr_hyperparams ))

        

    return results

def rank_grid_search_results(results, rank_by_metric, greater_is_better = True,return_top=True):
    try:
        _ = {x[0][rank_by_metric] for x in results}
    except: 
        raise ValueError("{0} error not found in at least some of the results".format(rank_by_metric))
    
    is_metric = -1 if greater_is_better else 1
    sorted_results = sorted(results, key = lambda x: is_metric*x[0][rank_by_metric])
    if return_top:
        return sorted_results[0] 
    else:
        return sorted_results


def fit_and_transform(train,val,transformation,numeric_features = [], categorical_features = []):
    features = numeric_features + categorical_features
    if transformation == 'none':
        return train.loc[:,features], val.loc[:,features]
    elif transformation == 'std':
        scalar = StandardScaler()
        scalar.fit(train[numeric_features])
        X_train = pd.concat([pd.DataFrame(scalar.transform(train[numeric_features]),columns = numeric_features)\
                                    , train[categorical_features].reset_index(drop=True)],axis=1)
        X_val = pd.concat([pd.DataFrame(scalar.transform(val[numeric_features]),columns = numeric_features)\
                        , val[categorical_features].reset_index(drop=True)],axis=1)
        return X_train, X_val

def impute_missing(train,val,missing_val_transformation,numeric_features = [], categorical_features =[]):
    if missing_val_transformation == 'none':
        return train,val
    elif missing_val_transformation == 'fill0':
        train[numeric_features] = train[numeric_features].fillna(0)
        val[numeric_features] = val[numeric_features].fillna(0)
        train[categorical_features] = train[categorical_features].fillna('missing')
        val[categorical_features] = val[categorical_features].fillna('missing')
        return train, val



def final_model_obj(train, test, best_params, y_var,numeric_features = [], categorical_features = [], weights = None):

    # predict using best params 
    # points of note :
    # 1) We combine the test and val. Now that we've narrowed down to the best params, we don't need that seperation anymore and can add the info in val to the train set 
    # 2) We should not use the test set for anything. Especially stuff like early stopping in XGBoost. (I'm made that error before)
    # 3) For any model where you have an early stop like argument and use the val set during training to find how many trees/estimators to build,
    # for such models ensure that get the optimal ntrees/estimators from training. 
    
    # If no weights are passed, set all obs to equal weight 
    if weights is None:
        weights = np.ones(train.shape[0])

    model_type = best_params['type']
    _transform = best_params['transformations']
    missing_val_impute = best_params[ 'missing_values']
    model_params = {key:best_params[key] for key in best_params.keys() if key not in ['type','transformations','missing_values']}

    X_train, _ = fit_and_transform(train,test,transformation = _transform,numeric_features = numeric_features, categorical_features = categorical_features)
    X_train, _ = impute_missing(X_train, _, missing_val_transformation=missing_val_impute,numeric_features=numeric_features, categorical_features=categorical_features)
    if model_type == 'xgboost':

        # Model specific params 
        num_boost_rounds = model_params['num_boost_round']
        model_params = {key:model_params[key] for key in model_params.keys() if key not in ['num_boost_round']}

        dmat_train = xgb.DMatrix(X_train, label = train[y_var], feature_names = X_train.columns, weight = weights)
        final_model = xgb.train(model_params, dtrain = dmat_train,
                            num_boost_round=num_boost_rounds,
                            verbose_eval=10)


    elif model_type =='xgb_sklearn':
        final_model = xgb.XGBRegressor(random_state = 42,**model_params)
        final_model.fit(X_train, train[y_var],
                verbose=True)

    elif model_type == 'elasticnet':
        final_model = ElasticNet(random_state = 42,**model_params)
        final_model.fit(X_train,train[y_var])

    elif model_type == 'poisson':
        final_model = PoissonRegressor(**model_params)
        final_model.fit(X_train, train[y_var])

    elif model_type == 'gamma':
        final_model = GammaRegressor(**model_params)
        final_model.fit(X_train, train[y_var])

    return final_model
    

def predict_and_score(m, train, test,y_var, best_params, metrics = ['forecasting_score','mae','r2','rmse','forecasting_report'],\
    numeric_features = [], categorical_features = [],verbose = True):

    X_train, X_test = fit_and_transform(train,test,transformation = best_params['transformations'],numeric_features = numeric_features, categorical_features = categorical_features)
    X_train, X_test = impute_missing(X_train, X_test, missing_val_transformation=best_params['missing_values'],numeric_features=numeric_features, categorical_features=categorical_features)
    features = X_train.columns
    # base, base1 =X_train, X_test

    if best_params['type']=='xgboost':
        X_train = xgb.DMatrix(X_train, label = train[y_var], feature_names = features)
        X_test = xgb.DMatrix(X_test, label = test[y_var], feature_names = features)


    # Scores for train
    predictions_test = m.predict(X_test)
    error_metrics_train = error_metrics(train[y_var], m.predict(X_train), metrics=metrics)
    error_metrics_test = error_metrics(test[y_var], predictions_test, metrics=metrics)
    if verbose:
        for key in error_metrics_train.keys():
            print('Train {0} : {1}'.format(key, error_metrics_train[key]))
        for key in error_metrics_test.keys():
            print('Test {0} : {1}'.format(key, error_metrics_test[key]))
    
    return predictions_test


def train_test_val_split(data,grouping, datetime_col, val_months = 2, test_months = 1, train_remaining_months = True):

    remove_datetime_col = False
    grouping = grouping if isinstance(grouping, list) else [grouping]
    
    if is_datetime(data[datetime_col]) == False:
        temp_col_name = datetime_col + '_' + str(np.random.choice(int(1e6)))
        data[temp_col_name] = pd.to_datetime(data[datetime_col])
        datetime_col = temp_col_name
        remove_datetime_col = True

    # Sort on datetime col + any grouping columns that may have been passed
    sort_grp = grouping + [datetime_col]
    sort_order = [True]*(len(sort_grp)-1) + [False]
    data = data.sort_values(by = sort_grp, ascending = sort_order)

    # Create a temporary rank column
    temp_rank_col = 'rank_' + str(np.random.choice(int(1e6)))
    data[temp_rank_col] = data.groupby(grouping)[datetime_col].rank(ascending = False)

    # Split into train, val and test months 
    columns_to_remove = [temp_rank_col, datetime_col] if remove_datetime_col else [temp_rank_col]
    train_df = data[data[temp_rank_col]>(val_months + test_months)].drop(columns = columns_to_remove)
    val_df = data[(data[temp_rank_col]<=(val_months + test_months))&(data[temp_rank_col]>test_months)].drop(columns = columns_to_remove)
    test_df = data[data[temp_rank_col]<=test_months].drop(columns = columns_to_remove)

    return train_df, val_df, test_df















# def error_by_model_hyperparams(results):
#     """Rank grid search results across all model types tested
#     Additionally, return a scores across hyperparams of each model type seperately"""
#     sorted_results = sorted(results, key = lambda x: is_metric*x[0][rank_by_metric])
#     model_types = list(set([x[1]['type'] for x in sorted_results]))
#     model_type_params = []
#     for _type in model_types:
#         single_type = []
#         for x in sorted_results:
#             if x[1]['type']==_type:
#                 single_type.append(x)
        
#         model_type_params.append(convert_to_df(single_type))

# def convert_to_df(list_errors):
#     """Convert a list of errors into a dataframe"""
#     errors = pd.DataFrame([[x[0]] for x in list_errors],columns = ['errors'])
#     values = pd.DataFrame([x[1].values() for x in list_errors], columns = list(list_errors[0][1].keys()))
#     return pd.concat([errors, values],axis=1)