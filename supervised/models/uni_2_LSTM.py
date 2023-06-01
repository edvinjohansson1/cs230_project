from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Activation

from preprocessing import embedding_layer_glove


def Uni_2_LSTM(input_shape, word_to_vec_map, word_to_index, lstm_units=[128, 128], dropout_rates=[0.5, 0.5]):
    """
    Implements a two-layer unidirectional LSTM with dropout after each
    LSTM layer as a Keras model object.
    
    Inputs:
        input_shape:     shape of input review, put it as (FLAGS.max_review_word_count, )
        word_to_vec_map: dictionary mapping from word to 100-dim vector embedding
        word_to_index:   dictionary mapping from word to index in vocabulary
        lstm_units:      list with number of units used in each LSTM layer
        dropout_rates:   list with dropout rates used in each dropout layer
    
    Outputs:
        model: A keras model object
    """
        
    # Input layer
    word_indices = Input(shape=input_shape, dtype='int32')
    
    # Embedding layer (pretrained with GloVe-100)
    embeddings = embedding_layer_glove(word_to_vec_map, word_to_index)(word_indices)
    
    # First LSTM layer and dropout
    X = LSTM(units=lstm_units[0], return_sequences=True)(embeddings)
    X = Dropout(rate=dropout_rates[0])(X)
    
    # Second LSTM layer
    X = LSTM(units=lstm_units[1], return_sequences=True)(X)
    X = Dropout(rate=dropout_rates[1])(X)
    
    # Intermediate dense layer followed by
    # dense layer with one output unit for prediction
    X = Dense(units=32, activation='relu')(X)
    X = Dense(units=1, activation='sigmoid')(X)
    
    
    # Finally, create the model object
    model = Model(inputs=word_indices, outputs=X, name='uni_2_LSTM')
    
    return model    