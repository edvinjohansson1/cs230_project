from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, ReLU, MaxPooling2D, Reshape, Flatten

from preprocessing import embedding_layer_glove


def CNN(input_shape, word_to_vec_map, word_to_index, FLAGS):
    """
    Implements a two-layer unidirectional LSTM with dropout after each
    LSTM layer as a Keras model object.
    
    Inputs:
        input_shape:     shape of input review, put it as (FLAGS.max_review_word_count, )
        word_to_vec_map: dictionary mapping from word to 100-dim vector embedding
        word_to_index:   dictionary mapping from word to index in vocabulary
        FLAGS:           hyperparameter settings
    
    Outputs:
        model: A keras model object
    """
    # b = batch_size
    # e = embedding_dim = 100
    
    # Input layer
    word_indices = Input(shape=input_shape, dtype='int32')                                        # (b, 200)
    
    # Embedding layer (pretrained with GloVe-100)
    embeddings = embedding_layer_glove(word_to_vec_map, word_to_index)(word_indices)              # (b, 200, 100)
    # Add 'channel' dimension of 1
    embeddings_w_channels = Reshape((embeddings.shape[1], embeddings.shape[2], 1))(embeddings)    # (b, 200, 100, 1)

    
    ########### CNN part (start) #######################
    Z1 = Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='valid')(embeddings_w_channels) # (b, 198, 98,  8)
    A1 = ReLU()(Z1)                                                                               # (b, 198, 98,  8)
    P1 = MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same')(A1)                       # (b,  99, 33,  8)
    
    Z2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='valid')(P1)                   # (b,  97, 31, 16)
    A2 = ReLU()(Z2)                                                                               # (b,  97, 31, 16)
    P2 = MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same')(A2)                       # (b,  49, 11, 16)
    
    Z3 = Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='valid')(P2)                   # (b,  45,  7, 32)
    A3 = ReLU()(Z3)                                                                               # (b,  45,  7, 32)
    P3 = MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same')(A3)                       # (b,  23,  3, 32)
    ########### CNN part (end) #########################
    
    F = Flatten()(P3)                                                                             # (b, 2208)
    out = Dense(units=1, activation='sigmoid')(F)                                                 # (b, 1)
    
    
    # Finally, create the model object
    model = Model(inputs=word_indices, outputs=out, name='CNN')
    
    return model