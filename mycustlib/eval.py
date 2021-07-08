import os
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error,\
                             mean_absolute_error,\
                             classification_report,\
                             recall_score,\
                             accuracy_score,\
                             f1_score,\
                             log_loss,\
                             precision_score)


class RegressionEvaluation:
 
 '''
 
 Constructors used to serialize model performance for RegressionEvaluation
 
 '''
 

 def __init__(self,precision=2):
    self.dataframe = pd.DataFrame(columns=['model',\
                                           'r_squared',\
                                           'train_mse',\
                                           'train_mae',\
                                           'train_lmae',\
                                           'val_mse',\
                                           'val_mae',\
                                           'val_lmae'])
    self.precision=precision 

 def get_metrics(self):
    '''
    Returns the current stored results
    
    '''
    return self.dataframe


 def add_metrics(self,y_train,y_val,X_train,\
                  X_val,model,model_desc):
    '''
        Store metrics within a dataframe for comparison.
    '''
   
    x_train_pred = model.predict(X_train)
    x_val_pred = model.predict(X_val)
        
    if model_desc in str(self.dataframe['model']):
        self.dataframe = self.dataframe[self.dataframe['model'] != model_desc]

    r_squared = round(model.score(X_train,y_train),self.precision)
    train_mse = round(mean_squared_error(y_train,x_train_pred),self.precision)
    train_mae = round(mean_absolute_error(y_train,x_train_pred),self.precision)
    train_lmae = round(pow(10,(np.sqrt(np.square(np.log10(x_train_pred +1) - np.log10(y_train +1)).mean()))),self.precision)
        
    val_mse = round(mean_squared_error(y_val,x_val_pred),self.precision)
    val_mae = round(mean_absolute_error(y_val,x_val_pred),self.precision)
    val_lmae = round(pow(10,np.sqrt(np.square(np.log10(x_val_pred +1) - np.log10(y_val +1)).mean())),self.precision)

    self.dataframe = (self.dataframe.append(pd.Series([model_desc,r_squared,train_mse,train_mae,train_lmae,\
                                         val_mse,val_mae,val_lmae],index =['model','r_squared','train_mse',\
                                               'train_mae','train_lmae','val_mse','val_mae','val_lmae'],),\
                                               ignore_index=True))

 def save_data(self,PROJECT_PATH):
      self.dataframe.to_csv(os.path.join(PROJECT_PATH,'eval','eval.csv'),
                            index=False)

 def load_data(self,path):
      self.dataframe = pd.read_csv(path)
      
      
      
      
      
#Custom class for storing evaluation metrics

class ClassificationEvaluation:

  def __init__(self,precision=2):
    self.dataframe = pd.DataFrame(columns=['model',\
                                           'train_accuracy',\
                                           'train_recall',\
                                           'train_precision','train_f1_score',\
                                           'train_log_loss','val_accuracy',\
                                           'val_recall','val_precision',\
                                           'val_f1_score','val_log_loss'])
    self.precision = 2

  def get_metrics(self):
    '''
    Returns the current stored results
    
    '''
    return self.dataframe

  def add_metrics(self,y_train,y_val,X_train,X_val,
                  model,model_desc,average='macro',
                  x_train_pred=[],x_val_pred=[],
                  x_train_pred_prob = [],
                  x_val_pred_prob = []):
    
    '''
    Store metrics within a dataframe for comparison.
    '''
    if len(x_train_pred) == 0:
      x_train_pred = model.predict(X_train)
    if len(x_val_pred) == 0:
      x_val_pred = model.predict(X_val)
    if len(x_train_pred_prob) == 0:
      x_train_pred_prob = model.predict_proba(X_train)
    if len(x_val_pred_prob) == 0:
      x_val_pred_prob = model.predict_proba(X_val)

    print("-------------------- Train Classification Report ----------------------\n")
    print(classification_report(y_true = y_train,y_pred = x_train_pred))
    print("-------------------- Test Classification Report ----------------------\n")
    print(classification_report(y_true = y_val,y_pred = x_val_pred))
    
    if model_desc in str(self.dataframe['model']):
      self.dataframe = self.dataframe[self.dataframe['model'] != model_desc]

    train_accuracy = accuracy_score(y_train,x_train_pred)
    train_recall = recall_score(y_train,x_train_pred,average=average)
    train_precision = precision_score(y_train,x_train_pred,average=average)
    train_f1_score = f1_score(y_train,x_train_pred,average=average)
    train_log_loss = log_loss(y_train,x_train_pred_prob)
    
    val_accuracy = accuracy_score(y_val,x_val_pred)
    val_recall = recall_score(y_val,x_val_pred,average=average)
    val_precision = precision_score(y_val,x_val_pred,average=average)
    val_f1_score = f1_score(y_val,x_val_pred,average=average)
    val_log_loss = log_loss(y_val,x_val_pred_prob)

    self.dataframe = self.dataframe.append(pd.Series([model_desc,train_accuracy,\
                                                      train_recall,\
                                                      train_precision,\
                                                      train_f1_score,\
                                                      train_log_loss,\
                                                      val_accuracy,val_recall,\
                                                      val_precision,\
                                                      val_f1_score,val_log_loss]
                                                    ,index =['model',\
                                                             'train_accuracy',\
                                                             'train_recall',
                                                             'train_precision',\
                                                             'train_f1_score',\
                                                             'train_log_loss',\
                                                             'val_accuracy',\
                                                             'val_recall',\
                                                             'val_precision',\
                                                             'val_f1_score',\
                                                             'val_log_loss'],),\
                                           ignore_index=True)

  def save_data(self,PROJECT_PATH):
      self.dataframe.to_csv(os.path.join(PROJECT_PATH,'eval','eval.csv'),
                            index=False)

  def load_data(self,path):
      self.dataframe = pd.read_csv(path)