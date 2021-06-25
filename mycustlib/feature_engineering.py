import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Embedding,Input,Conv1D,AveragePooling1D,BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

class GloveAutoEncoder:
  
  def __init__(self):
    self.t = Tokenizer()
    self.embedding_index = dict()
    self.embedding_vector = []
    self.embedding_matrix = []
    self.max_length = 4
    self.ae = Model()
    self.docs_encoded = []
    self.col_name = ''
    self.std = MinMaxScaler()
  
  def create_embedding_index(self,glove_module):
    print('Indexing word vector')
    f = open(glove_module,encoding='utf-8')
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:],dtype='float32')
      self.embedding_index[word] = coefs
    
    print(f'Indexed {len(self.embedding_index)} words')
    f.close()
    
    return self.embedding_index
  
  def create_embedding_matrix(self,vocab_size,\
                              dimension,\
                              embedding_index,\
                              word_index):
    self.embedding_matrix = np.zeros((vocab_size,dimension))
    for word,i in word_index.items():
      self.embedding_vector = embedding_index.get(word)
      if self.embedding_vector is not None:
        self.embedding_matrix[i] = self.embedding_vector
    print(f'Here is shape of embedding matrix {self.embedding_matrix.shape}')
    return self.embedding_matrix
  
  def create_encoder(self,max_length,\
                            vocab_size,\
                            dimension,\
                            embedding_matrix,\
                            encoding_dim,\
                    padded_docs,\
                    epochs,\
                    learning_rate):
    input = Input(shape=(max_length,),dtype='int32')
    embed = Embedding(input_dim=vocab_size,\
                      output_dim=dimension,\
                      weights=[embedding_matrix],\
                      trainable=False)(input)
    conv = Conv1D(8,3,activation='relu',padding='same')(embed)
    pool = AveragePooling1D(pool_size=3)(conv)
    flatten = Flatten()(pool)
    norm = BatchNormalization()(flatten)
    output = Dense(max_length,activation='linear')(norm)
    model = Model(input,output)
    model.compile(loss='mse',optimizer=RMSprop(learning_rate=learning_rate))
    print(model.summary())
    hist = model.fit(padded_docs,padded_docs,epochs=epochs,\
                     validation_data=(padded_docs,padded_docs))
    self.ae = model
    
    return self.ae
    
  def create_glove_encoding(self,\
                              docs,\
                              col_name,\
                              dimension=100,\
                              glove_module='glove.6B.100d.txt',\
                              max_length=4,\
                              encoding_dim = 50,\
                              epochs=200,\
                              learning_rate=.001):
    
    self.max_length = max_length
    self.col_name = col_name
    self.encoding_dim = encoding_dim
    self.t.fit_on_texts(docs)
    word_index = self.t.word_index
    encoded_docs = self.t.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs,\
                                  maxlen=max_length,\
                                  padding='post')
    print(f'There are {len(self.t.word_index)} words in data')
    vocab_size = len(self.t.word_index) + 1
    self.embedding_index = self.create_embedding_index(glove_module)
    self.embedding_matrix = self.create_embedding_matrix(vocab_size,\
                                                dimension,\
                                                self.embedding_index,\
                                                word_index)
    self.ae = self.create_encoder(max_length,\
                                vocab_size,\
                                dimension,\
                                self.embedding_matrix,\
                                encoding_dim,\
                          padded_docs,\
                          epochs,\
                          learning_rate)
    
    
    self.docs_encoded = self.ae.predict(padded_docs)
    self.std.fit(self.docs_encoded)
    self.docs_encoded = self.transform_glove_encoding(docs)
    
    
    return self.docs_encoded

  def transform_glove_encoding(self,docs):
    encoded_docs = self.t.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs,\
                                  maxlen=self.max_length,\
                                  padding='post')
    docs_encoded = self.ae.predict(padded_docs)
    cols = [ (self.col_name + str(i)) for i in range(docs_encoded.shape[1])]
    docs_encoded = self.std.transform(docs_encoded)
    docs_encoded = pd.DataFrame(docs_encoded,\
                                columns=cols,\
                                index=docs.index)
  
    return docs_encoded
    