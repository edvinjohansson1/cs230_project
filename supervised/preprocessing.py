import numpy as np
from keras.layers.embeddings import Embedding

def review_to_indices(X, word_to_index, max_review_word_count):
    """ Converts an array of reviews (each as one string) into an array of
        indices corresponding to words in the review. The output shape
        should be such that it can be given to a keras Embedding() layer. 
        
        Inputs:
            X: array of reviews of shape (m, 1)
            word_to_index: dictionary that maps a word to its index
            
        Outputs:
        X_indices: array of indices corresponding to words in the reviews
                   of X. shape (m, max_review_word_count)
    """
    
    m = X.shape[0]   # batch size
    
    X_indices = np.zeros((m, max_review_word_count))
    
    for i in range(m):
        review = X[i].lower().split(' ')

        for j, word in enumerate(review):
            # if word is not found in 
            X_indices[i, j] = word_to_index.get(word, 0) 
    
    return X_indices

def embedding_layer_glove(word_to_vec_map, word_to_index):
    """ Creates a keras Embedding() layer and loads the weights from
        pre-trained GloVe 100-dimensional vectors. 
        
        Inputs:
            
    """
    input_dim  = len(word_to_index) + 1
    output_dim = word_to_vec_map['a'].shape[0]
    
    embedding_matrix = np.zeros((input_dim, output_dim))
    
    for word, index in word_to_index.items():
        # set the embedding to all zeros if not found in the GloVe dictionary
        embedding_matrix[index, :] = word_to_vec_map.get(word, np.zeros(output_dim))
        
    # probably not worth it to train the embedding parameters further
    embedding_layer = Embedding(input_dim=input_dim,
                                output_dim=output_dim,
                                trainable=False)
    
    # Shape (None, ), to allow for arbitrary input length
    embedding_layer.build(input_shape=(None, ))
    
    # Set weights to the weights from GloVe-100
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer