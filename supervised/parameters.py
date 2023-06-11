import argparse

# Parameters
# ==================================================
parser = argparse.ArgumentParser()
# parser.add_argument('--flag_name', type=str, default='default_value', help='Description of the flag')

# Data loading params
parser.add_argument('--dev_sample_percentage', type=float, default=.05, help='Percentage of the training data to use for validation (default: 0.05)')
# parser.add_argument('--data_file',             type=str,  default='/home/ubuntu/cs230_project/data/amazon_reviews_us_Electronics_v1_00.tsv', help='Data source for review')
# parser.add_argument('--word2vec_file',         type=str,  default='/home/ubuntu/cs230_project/data/glove.6B.100d.txt', help='File path for word2vec')
parser.add_argument('--data_file',             type=str,  default='/content/drive/MyDrive/CS230/Project/git/cs230_project/data/amazon_reviews_us_Electronics_v1_00.tsv', help='Data source for review')
parser.add_argument('--data_file2',             type=str,  default='/content/drive/MyDrive/CS230/Project/git/cs230_project/data/amazon_reviews_us_Kitchen_v1_00.tsv', help='Data source for review')
parser.add_argument('--word2vec_file',         type=str,  default='/content/drive/MyDrive/CS230/Project/git/cs230_project/data/glove.6B.100d.txt', help='File path for word2vec')

# Text parsing parameters
parser.add_argument('--min_total_votes', type=int, default=5, help='Min amount of total votes on a review for it to be used (default: 10)')
parser.add_argument('--max_review_word_count', type=int, default=200, help='Max amount of words for a review before it is discarded (default: 256)')
parser.add_argument('--keep_start_of_longer_reviews', type=bool, default=True, help="True: reviews longer than 'max_review_word_count' are cut off and used. False: reviews longer are discarded completly (default: True)")

# Training parameters
parser.add_argument('--batch_size',       type=int, default=32,  help='Batch Size (default: 32)')
parser.add_argument('--num_epochs',       type=int, default=50,  help='Number of training epochs (default: 20)')

# Misc.
parser.add_argument('--debug_mode', type=bool, default=False, help='Set when running in debug mode to skip certain parts of the code (defualt: False)')

FLAGS = parser.parse_args('')
print("\nParameters:")
for attr, value in vars(FLAGS).items():
    print(f'\t{attr}: {value}')