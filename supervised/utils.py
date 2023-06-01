import numpy as np
import csv

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def read_glove_vecs(glove_file):
    """ Read a GloVe with embeddings for a dictionary of words and outputs the following:
        
        outputs:
            words_to_index:  dictionary that maps from words to indices
            index_to_words:  dictionary that maps from indices to words
            word_to_vec_map: dictionary that maps from words to embeddings
    """
    print('\nCreating word embeddings matrix ...')
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    
    print('Done')
    
    return words_to_index, index_to_words, word_to_vec_map

def load_data(chunk_path, min_total_votes, max_review_word_count, keep_start_of_longer_reviews):
    x_text = []
    y = []
    count = 0 # only used for printing purposes at the end
    
    with open(chunk_path, 'r') as tsv_chunk_file:
        print("\nLoading data ...")
        reader = csv.reader(tsv_chunk_file, delimiter='\t')
        header = next(reader)
        
        # Convert the header list to a dictionary for easier access
        header_dict = {column: index for index, column in enumerate(header)}

        for row in reader:
            review_body   = row[header_dict['review_body']]
            # star_rating   = row[header_dict['star_rating']]
            helpful_votes = int(row[header_dict['helpful_votes']])
            total_votes   = int(row[header_dict['total_votes']])
            
            # Only include reviews with enough votes
            if total_votes < min_total_votes:
                continue
            
            # check if length is short enough
            review_body_split = review_body.split(" ")
            if len(review_body_split) > max_review_word_count:
                if not keep_start_of_longer_reviews:
                    continue
            
            review_body = ' '.join(review_body_split[:max_review_word_count])
            x_text.append(review_body)
            y.append(helpful_votes/total_votes)
            count += 1
            
        print(f'Chunk loaded. Found {count} data points with >= {min_total_votes} total votes.')
        
        X = np.asarray(x_text)
        Y = np.asarray(y)
        
        return X, Y
