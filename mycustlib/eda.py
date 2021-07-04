#Custom functions for ease of use

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS


from wordcloud import WordCloud
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.express as px


def explain_data(dataframe):
  ''' 
  Returns a dataframe with columns such as shape, data types,
  missing values and descriptive stats for all columns of input dataframe. 
  
  '''
  
  print(f"The data has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
  
  print('Below are the column wise data-types,missing values, unique level and descriptive stats of the data')
  
  dtypes = dataframe.dtypes
  missing_values = dataframe.isnull().sum()
  unique_values = dataframe.nunique()
  describe = dataframe.describe(include='all').transpose()
  details = pd.concat([dtypes,missing_values,unique_values],axis =1,\
                      keys=['dtypes','missing_values','unique_values'])
  details = pd.concat([details,describe],axis=1)
  return details



def get_multi_value_keys(dataframe,key,value):
  
  ''' 
  Identifying keys mapped to multiple values in a dataframe. 
  '''
  
  multivalue_keys = dataframe.groupby([key])[value].nunique()
  multivalue_keys = multivalue_keys[multivalue_keys>1].\
                    sort_values(ascending=False)
  print(f'\nThere are {len(multivalue_keys[multivalue_keys>1])} multivalue keys \n')
  return multivalue_keys
  
 
 
def bucketize_pareto(dataframe,id_column,target_column,pareto=.95,values=None,\
                     aggfunc=None):
  ''' 
   Identifies levels in id_column constituting pareto in target column for each 
   class in a dataframe. Returns the dataframe with list of pareto levels. 
   In case, there are values specify along with aggfunc.
  '''
  dataframe[id_column] = dataframe[id_column].astype('object')
  dataframe[target_column] = dataframe[target_column].astype('object')

  if values != None:
    values = dataframe[values]
  
  target_cross = pd.crosstab(dataframe[id_column],dataframe[target_column],\
                             values=values,aggfunc=aggfunc).reset_index().\
                             fillna(0)
  columns = target_cross.columns.difference([id_column])
  
  for i in columns:
    target_cross[i] = target_cross[i]/sum(target_cross[i])

  target_cross['agg']  = target_cross.loc[:,columns].mean(axis=1)   
  target_cross.sort_values(by=['agg'],inplace=True,ascending=False,\
                           ignore_index=True)
  target_cross['perc_contri'] = target_cross['agg']/sum(target_cross['agg'])
  target_cross['perc_cumu'] = 0
  
  for i in range(len(target_cross)):
    if i == 0:
      target_cross.loc[i,'perc_cumu'] = target_cross.loc[i,'perc_contri']
    else:
      target_cross.loc[i,'perc_cumu'] = target_cross.loc[i,'perc_contri'] +\
       target_cross.loc[i-1,'perc_cumu']

  ids = list(target_cross[target_cross['perc_cumu'] <= pareto][id_column])
  print(f'{len(ids)} ids identified in the Top {pareto*100}% pareto')

  dataframe['top_' + id_column] = (dataframe[id_column].\
                                   apply(lambda x: x if x in ids else 'other_'+\
                                         id_column))
  
  return (dataframe,ids)
  

  
def create_word_cloud (text,stop_words = [],\
                       min_sen_len = 2,\
                       max_font_size=80,\
                       max_words=30,\
                       background_color='red',\
                       figsize=(10,8)):
  
  ''' 
  Creates a word cloud of the provided series of sentences
  '''

  text = [s.lower() for s in text]
  all_stop_words = list(STOP_WORDS) + stop_words
  
    
  clean_text = []

  for i,e in enumerate(text):
    tokens = [t.lemma_ for t in nlp(e) if str(t) not in all_stop_words and t.is_alpha == True]
    s = " ".join(tokens)
    clean_text.append(s)
    

  clean_text = [i for i in clean_text if len(i) >= min_sen_len]
  wordcloud = WordCloud(max_font_size=max_font_size,\
                        max_words=max_words,\
                        background_color=background_color).\
                        generate(str(clean_text))
  
  plt.figure(figsize=figsize)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()
  
  return wordcloud

def plot_one_cat(dataframe,column):
  '''
  Plots a frequency distribution of a column provided in a dataframe via a bar 
  chart.Each bar is marked with percentage value out of 100 which is normalized
  distribution of column against the granularity of data shared. 

  '''
  if dataframe[column].dtype in ['int64','int32']:
    dataframe[column] = dataframe[column].astype('category')

  value_count = dataframe[column].value_counts(normalize=True)*100
  value = value_count.values.round(2)
  index = list(value_count.index)
  trace = go.Bar(y=value,x=index,text=value,textposition='auto')
  data = [trace]

  layout = go.Layout(title=f"{column} distribution",xaxis_title=column,\
                     xaxis_tickangle=90)
  figure = go.Figure(data=data,layout=layout)
  figure.show()
 
 
def plot_two_cat(dataframe,column_y,column_x):
  '''
  Plots a frequency distribution of a column_y on Y-axis while grouping by 
  levels in column_x across X-axis using a stacked bar chart denoting 
  percentage contribution of each Y in X against the granularity of data shared. 
  '''

  if dataframe[column_x].dtype in ['int64','int32']:
    dataframe[column_x] = dataframe[column_x].astype('category')

  if dataframe[column_y].dtype in ['int64','int32']:
    dataframe[column_y] = dataframe[column_y].astype('category')
   
  levels_x = sorted(list(dataframe[column_x].unique()))
  levels_y = list(dataframe[column_y].unique())
  data = []
  for i in levels_x:
      value_count = dataframe[column_y][dataframe[column_x]==i].value_counts()
      value = value_count.values
      index = list(value_count.index)
      trace = go.Bar(y=value,\
                     x=index,\
                     text=value,\
                     textposition='inside',\
                     name=i)
      data = data + [trace]

  layout = go.Layout(title=f"{column_y} vs {column_x} distribution",xaxis_title\
                     = column_y,barmode='stack',barnorm='percent',\
                     xaxis_tickangle=90)
  figure = go.Figure(data=data,layout=layout)
  figure.show()
 
def plot_explained_variance(pca,threshold=.95):
  ''' 
  plots explained variance in a bar chart for PCA
  '''
  y = np.round(pca.explained_variance_ratio_*100,2)
  x = np.arange(len(y))
  y_cumu = y.cumsum()
  y_thres = np.ones(len(y))*threshold*100


  fig = go.Figure()
  fig.add_trace(go.Bar(x=x, y=y,text=y,\
                       textposition='auto',name='% Explained Variance'))
  fig.add_trace(go.Scatter(x=x,y=y_cumu,\
                           text=y_cumu,name="Cumulative Explained Variance"))
  fig.add_trace(go.Scatter(x=x,y=y_thres,\
                           name='95% explained variance',mode='markers'))
  fig.show()