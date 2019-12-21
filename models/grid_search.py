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
        self.EMBED_DIM = 100
        self.N_CLASSES = 2
        self.ONLINE_SOURCES = True
        self.TRAIN_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/train-data-improved-sentiwordnet-arguingfull.tsv?token=AD7GEDLFTVHGUIDOG4EDKYK57FJJY'
        self.DEV_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/dev-improved-sentiwordnet-arguingfull.tsv?token=AD7GEDKHMNRQLNNRBNDYWJK57FJJ6'
        self.EMBEDDING_PATH = 'gdrive/My Drive/colab_projects/data/glove.6B.100d.txt'
        self.UNCASED = True  # If true, words are turned into lower case.

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
    if len(max_entry) == 1:
        return max_entry[0]
    if 'I' in max_entry:
        return 'I'
    if 'B' in max_entry:
        return 'B'
    return max_entry[0]


def run_config(config, file_prefix, data=None, repetitions=5,
               majority_voting=False, verbose=True):
    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    predictions = None
    label_cols = []
    for i in range(repetitions):
        if verbose:
            print("Iteration " + str(i + 1) + " of " + str(repetitions))
        data, labels = run(config, data=data, verbose=verbose,
                           file_prefix=file_prefix, file_stem=now,
                           file_suffix=str(i))
        if majority_voting:
            if predictions is None:
                predictions = labels
                predictions = predictions.rename(columns={'label': 'label_0'})
            else:
                predictions.insert(loc=len(predictions.columns),
                                   column='label_' + str(i),
                                   value=labels.label)
            label_cols.append('label_' + str(i))
    if majority_voting:
        labels = []
        for row in predictions.itertuples():
            labels.append(get_majority_vote(
                [getattr(row, l) for l in label_cols]))
        predictions['label'] = labels
        spans = si_predictions_to_spans(predictions)
        print_spans(spans, file_prefix, now, 'majority')

    # Return data in case the next config only changes model features
    return data


file_prefix = '/content/gdrive/My Drive/semeval-predictions/'
data = None

# for epochs in [5]:
#     for dropout in [0.2, 0.4]:
#         config = Config({'EPOCHS': epochs, 'DROPOUT': dropout})
#         data = run_config(config, file_prefix, data)

# config = Config()
config = Config({'TRAIN_URL': 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/train-data-improved-sentiwordnet-arguingfull-pos.tsv?token=AD7GEDNDI6GENLIHFDOMX4K6ASWAU',
                 'DEV_URL': 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/dev-improved-sentiwordnet-arguingfull-pos.tsv?token=AD7GEDJPX4IDUX7OOWTHD7S6ASWBQ'})
data = run_config(config, file_prefix, data, majority_voting=True)
