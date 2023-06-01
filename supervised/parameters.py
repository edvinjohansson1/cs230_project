import argparse

# Parameters
# ==================================================
parser = argparse.ArgumentParser()
# parser.add_argument('--flag_name', type=str, default='default_value', help='Description of the flag')

# Data loading params
parser.add_argument('--dev_sample_percentage', type=float, default=.05, help='Percentage of the training data to use for validation (default: 0.05)')
parser.add_argument('--data_file',             type=str,  default='/home/ubuntu/cs230_project/data/amazon_reviews_us_Electronics_v1_00.tsv', help='Data source for review')
parser.add_argument('--word2vec_file',         type=str,  default='/home/ubuntu/cs230_project/data/glove.6B.100d.txt', help='File path for word2vec')

# # Model Hyperparameters
# parser.add_argument('--embedding_dim',         type=int,   default=100, help='Dimensionality of character embedding (default: 100)')
# parser.add_argument('--filter_sizes',          type=str,   default='3,4,5', help="Comma-separated filter sizes (default: '3,4,5')")
# parser.add_argument('--num_filters',           type=int,   default=128, help='Number of filters per filter size (default: 128)')
# parser.add_argument('--dropout_keep_prob',     type=float, default=0.5, help='Dropout keep probability (default: 0.5)')
# parser.add_argument('--l2_reg_lambda',         type=float, default=0.5, help='L2 regularization lambda (default: 0.5)')

# Text parsing parameters
parser.add_argument('--min_total_votes', type=int, default=10, help='Min amount of total votes on a review for it to be used (default: 10)')
parser.add_argument('--max_review_word_count', type=int, default=200, help='Max amount of words for a review before it is discarded (default: 256)')
parser.add_argument('--keep_start_of_longer_reviews', type=bool, default=True, help="True: reviews longer than 'max_review_word_count' are cut off and used. False: reviews longer are discarded completly (default: True)")

# Training parameters
parser.add_argument('--batch_size',       type=int, default=64,  help='Batch Size (default: 64)')
parser.add_argument('--num_epochs',       type=int, default=20,  help='Number of training epochs (default: 20)')
# parser.add_argument('--evaluate_every',   type=int, default=100, help='Evaluate model on dev set after this many steps (default: 100)')
# parser.add_argument('--checkpoint_every', type=int, default=100, help='Save model after this many steps (default: 100)')
# parser.add_argument('--num_checkpoints',  type=int, default=5,   help='Number of checkpoints to store (default: 5)')

# # Misc Parameters
# parser.add_argument('--allow_soft_placement', type=bool, default=True, help='Allow device soft device placement (default: True')
# parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices (default: False)')

parser.add_argument('--debug_mode', type=bool, default=False, help='Set when running in debug mode to skip certain parts of the code (defualt: False)')

FLAGS = parser.parse_args('')
print("\nParameters:")
for attr, value in vars(FLAGS).items():
    print(f'\t{attr}: {value}')