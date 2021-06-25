import os
import pickle
import pandas as pd 
import plotly.express as px
from sklearn.model_selection import GridSearchCV

def load_model(PROJECT_PATH,model_desc):
  '''
  Executes steps to load model
  ''' 
  model_name = str(model_desc) + '.pkl'
  #Save model
  model = pickle.load(open(os.path.join(PROJECT_PATH,\
                                      'model',model_name),'rb'))

  return model


def build_model(model,
                params,
                X_train,
                y_train,
                X_val,
                model_desc,
                PROJECT_PATH,
                sample_weight=None,
                scoring=None,
                verbose=1,
                has_sample_weight=1):
  '''
  Executes steps to train model and save model
  ''' 
  gs = GridSearchCV(model,param_grid=params,scoring=scoring,n_jobs=-1,\
                    verbose=verbose)
  if has_sample_weight ==1:
    gs.fit(X_train,y_train,sample_weight=sample_weight)
  else:
    gs.fit(X_train,y_train)
  print(f'Best parameters for gs are {gs.best_params_}')
  model = gs.best_estimator_
  x_train_pred = model.predict(X_train)
  x_val_pred = model.predict(X_val)
  print(f'Here are the Top 100 train results \n{x_train_pred[:100]}')
  print(f'Here are the Top 100 val results \n{x_val_pred[:100]}')

  model_name = str(model_desc) + '.pkl'
  #Save model
  pickle.dump(model,open(os.path.join(PROJECT_PATH,\
                                      'model',model_name),'wb'))

  return model

def plot_feature_importances(dataframe,model):
  '''
  Plots feature importances for tree based models.
  Needs dataframe for feature names
  '''
  column_name = pd.Series(dataframe.columns)
  importances = pd.Series(model.feature_importances_)
  feature_importances = pd.concat([column_name,\
                                   importances],axis=1,\
                                  keys=['feature','importance'])
  feature_importances.sort_values('importance',\
                                  ascending=False,\
                                  inplace=True,ignore_index=True)
  figure = px.bar(feature_importances,x='feature',y='importance')
  figure.show()
  return feature_importances