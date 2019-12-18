from model import run
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


def run_config(config, file_prefix, data=None, repetitions=5, verbose=True):
    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    for i in range(repetitions):
        if verbose:
            print("Iteration " + str(i + 1) + " of " + str(repetitions))
        data = run(config, data=data, verbose=verbose,
                   file_prefix=file_prefix, file_stem=now, file_suffix=str(i))
    # Return data in case the next config only changes model features
    return data


file_prefix = '/content/gdrive/My Drive/semeval-predictions/'
data = None
for epochs in [5]:
    for dropout in [0.2, 0.4]:
        config = Config({'EPOCHS': epochs, 'DROPOUT': dropout})
        data = run_config(config, file_prefix, data)
