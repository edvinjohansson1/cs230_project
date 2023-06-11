from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional

from preprocessing import embedding_layer_glove


def BiLSTM(input_shape, word_to_vec_map, word_to_index, lstm_units=[64, 32], dropout_rates=[0.5, 0.5]):
    """
    Implements a two-layer bi-directional LSTM with dropout after each
    LSTM layer as a Keras model object.
    
    Inputs:
        input_shape:     shape of input review, put it as (FLAGS.max_review_word_count, )
        word_to_vec_map: dictionary mapping from word to 100-dim vector embedding
        word_to_index:   dictionary mapping from word to index in vocabulary
        lstm_units:      list with number of units used in each LSTM layer 
    
    Outputs:
        model: A keras model object
    """
        
    # Input layer
    word_indices = Input(shape=input_shape, dtype='int32')
    
    # Embedding layer (pretrained with GloVe-100)
    embeddings = embedding_layer_glove(word_to_vec_map, word_to_index)(word_indices)
    
    # First LSTM layer and dropout
    X = Bidirectional(LSTM(units=lstm_units[0], return_sequences=True, recurrent_regularizer='l2'))(embeddings)
    X = Dropout(rate=dropout_rates[0])(X)
    
    # Second LSTM layer
#     X = Bidirectional(LSTM(units=lstm_units[1], return_sequences=True))(X)
    X = Bidirectional(LSTM(units=lstm_units[1], recurrent_regularizer='l2'))(X)
    X = Dropout(rate=dropout_rates[1])(X)
    
    # Intermediate dense layers followed by
    # dense layer with one output unit for prediction
#     X = Dense(units=64, activation='relu')(X)
#     X = Dense(units=16, activation='relu')(X)
    X = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(X)
    
    
    # Finally, create the model object
    model = Model(inputs=word_indices, outputs=X, name='biLSTM')
    
    return model 