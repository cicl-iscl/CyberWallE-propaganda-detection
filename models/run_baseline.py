from baseline_lstm import run
from google.colab import drive


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
               'O weight: ' + str(config.O_WEIGHT) + \
               ', I weight:' + str(config.I_WEIGHT) + \
               ', B weight: ' + str(config.B_WEIGHT) + '\n' + \
               'hidden units: ' + str(config.LSTM_UNITS) + '\n' + \
               'dropout rate: ' + str(config.DROPOUT) + '\n' + \
               'optimizer: ' + config.OPTIMIZER + '\n' + \
               'metric: ' + config.METRIC + '\n' + \
               'loss: ' + config.LOSS + '\n'


drive.mount('/content/gdrive')
config = Config()
file_prefix = '/content/gdrive/My Drive/'
data, model, history = run(config, file_prefix=file_prefix)
config.DROPOUT = 0.5
config.EPOCHS = 14
run(config, data=data, file_prefix=file_prefix)
