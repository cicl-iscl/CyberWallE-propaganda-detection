from model import run, si_predictions_to_spans, print_spans
from collections import Counter
import time


class Config:
    def __init__(self, args=None):
        """Creates a default configuration.

        Keyword arguments:
        args -- a dict(str -> ?) containing values diverging from the default
        """
        # Encoding the data:
        self.MAX_SEQ_LEN = 35
        self.EMBED_DIM = 300
        self.N_CLASSES = 2
        self.ONLINE_SOURCES = True
        self.TRAIN_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/train-improved-sentiwordnet-arguingfullindiv-pos.tsv?token=AD7GEDKUIWCZYSV6NDSU5JK6D5IPW'
        self.DEV_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/dev-improved-sentiwordnet-arguingfullindiv-pos.tsv?token=AD7GEDO6NHWP2COYFHNU7BK6D4APA'
        self.EMBEDDING_PATH = 'gdrive/My Drive/colab_projects/data/glove.42B.300d.zip'  # 'gdrive/My Drive/colab_projects/data/glove.6B.100d.zip'
        self.UNCASED = True  # If true, words are turned into lower case.
        self.SAVE_DATA = False  # If true, the following two values can be used
                                # for re-using the data next time.
        # In case the training & dev data were saved and can be reused:
        self.DATA_PATH = 'gdrive/My Drive/colab_projects/data/data/'
        self.LOAD_DATA = False

        # Building the model:
        self.BATCH_SIZE = 128
        self.EPOCHS = 10
        self.O_WEIGHT = 1.0
        self.I_WEIGHT = 6.5
        self.B_WEIGHT = 6.5
        self.LSTM_UNITS = 512
        self.DROPOUT = 0.25
        self.OPTIMIZER = 'adam'
        self.METRIC = 'categorical_accuracy'
        self.LOSS = 'categorical_crossentropy'

        # Making predictions:
        self.MAJORITY_VOTING = True

        if args:
            for key in args:
                setattr(self, key, args[key])

    def pretty_str(self):
        return 'max seq len: ' + str(self.MAX_SEQ_LEN) + '\n' + \
               'embedding depth: ' + str(self.EMBED_DIM) + '\n' + \
               'number of labels: ' + str(config.N_CLASSES) + '\n' + \
               'batch size: ' + str(self.BATCH_SIZE) + '\n' + \
               'epochs: ' + str(self.EPOCHS) + '\n' + \
               'O weight: ' + str(self.O_WEIGHT) + \
               ', I weight:' + str(self.I_WEIGHT) + \
               ', B weight: ' + str(self.B_WEIGHT) + '\n' + \
               'hidden units: ' + str(self.LSTM_UNITS) + '\n' + \
               'dropout rate: ' + str(self.DROPOUT) + '\n' + \
               'optimizer: ' + self.OPTIMIZER + '\n' + \
               'metric: ' + self.METRIC + '\n' + \
               'loss: ' + self.LOSS + '\n'


def get_majority_vote(votes):
    votes = dict(Counter(votes))
    max_count = -1
    max_entry = []
    for key in votes:
        count = votes[key]
        if count > max_count:
            max_count = count
            max_entry = [key]
        elif count == max_count:
            max_entry.append(key)
    # For our data, preferring specific labels in tie situations actually
    # doesn't make a difference.
    return max_entry[0]


def run_config(config, file_prefix, data=None, repetitions=5, verbose=True):
    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    predictions = None
    label_cols = []
    for i in range(repetitions):
        if verbose:
            print("Iteration " + str(i + 1) + " of " + str(repetitions))
        data, labels = run(config, data=data, verbose=verbose,
                           file_prefix=file_prefix, file_stem=now,
                           file_suffix=str(i))
        if config.MAJORITY_VOTING:
            if predictions is None:
                predictions = labels
                predictions = predictions.rename(columns={'label': 'label_0'})
            else:
                predictions.insert(loc=len(predictions.columns),
                                   column='label_' + str(i),
                                   value=labels.label)
            label_cols.append('label_' + str(i))
    if config.MAJORITY_VOTING:
        labels = []
        for row in predictions.itertuples():
            labels.append(get_majority_vote(
                [getattr(row, l) for l in label_cols]))
        predictions['label'] = labels
        spans = si_predictions_to_spans(predictions)
        print_spans(spans, file_prefix, now, 'majority')

    # Return data in case the next config only changes model features
    return data


file_prefix = '/content/gdrive/My Drive/colab_projects/semeval-predictions/'
data = None

# for epochs in [5, 10, 15]:
#     for dropout in [0.2, 0.4, 0.6, 0.8]:
#         config = Config({'EPOCHS': epochs, 'DROPOUT': dropout})
#         data = run_config(config, file_prefix, data)

# You can change config values by passing a dictionary to the constructor:
# config = Config()
# config = Config({'SAVE_DATA': True})
config = Config({'LOAD_DATA': True})
data = run_config(config, file_prefix, data)
