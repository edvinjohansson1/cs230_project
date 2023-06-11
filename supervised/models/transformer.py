from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Embedding
from keras.layers import MultiHeadAttention, LayerNormalization, Layer
from keras import Sequential
from tensorflow import range as tf_range
from tensorflow import shape

from preprocessing import embedding_layer_glove

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,
                 word_to_vec_map, word_to_index):
        super().__init__()
        self.token_emb = embedding_layer_glove(word_to_vec_map, word_to_index)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = shape(x)[-1]
        positions = tf_range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rates=[0.2, 0.2]):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation='relu'),
                               Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rates[0])
        self.dropout2 = Dropout(rates[1])

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def Transformer(input_shape, word_to_vec_map, word_to_index,
                dropout_rates_transformer=[0.2, 0.2], num_heads=3, ff_dim=32,
                dropout_rates_dense=[0.2, 0.1]):
    """
    Implements a Transformer as a Keras model object.
    
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
    embeddings = TokenAndPositionEmbedding(maxlen=input_shape[0],
                                           vocab_size=len(word_to_index)+1,
                                           embed_dim=100,
                                           word_to_vec_map=word_to_vec_map,
                                           word_to_index=word_to_index)\
                                           (word_indices)

    # Transformer layer
    X = TransformerBlock(embed_dim=100,
                         num_heads=num_heads,
                         ff_dim=ff_dim,
                         rates=dropout_rates_transformer)(embeddings)
    
    # Average the output from all time steps and apply two dense layers 
    # with a sigmoid at the end
    X = GlobalAveragePooling1D()(X)
    X = Dropout(rate=dropout_rates_dense[0])(X)
    X = Dense(units=16, activation='relu', kernel_regularizer='l2')(X)
    X = Dropout(rate=dropout_rates_dense[1])(X)
    X = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(X)
    
    # Finally, create the model object
    model = Model(inputs=word_indices, outputs=X, name='transformer')
    
    return model