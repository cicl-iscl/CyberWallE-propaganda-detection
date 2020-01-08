import pandas as pd
import numpy as np
from itertools import takewhile
import zipfile
import urllib.request
from keras.layers import Bidirectional, CuDNNLSTM, Dense, Dropout, \
    TimeDistributed
from keras.models import Sequential

########################
# Processing the input #
########################


# Helper method for prepare_data
def get_comments(filename, url=True):
    if url:
        comments = []
        with urllib.request.urlopen(filename) as f:
            for line in f:
                if line.startswith(b'#'):
                    comments.append(line.decode("utf-8"))
                else:
                    break
        return comments
    with open(filename, 'r', encoding='utf8') as f:
        commentiter = takewhile(lambda s: s.startswith('#'), f)
        comments = list(commentiter)
    return comments


# Helper method for prepare_data
def get_cols(input_df, col):
    return input_df.groupby('sent_id')[col].apply(list).to_frame()


# Helper method for prepare_data
def add_sent_lens(input_df, col='token'):
    input_df['n_toks'] = input_df[col].apply(lambda x: len(x))
    return input_df


# Helper method for prepare_data
def get_features(input_df, feature_cols):
    x = add_sent_lens(get_cols(input_df, 'token'))
    for feature in feature_cols:
        x = pd.merge(left=x, right=get_cols(input_df, feature),
                     left_on='sent_id', right_on='sent_id')
    return x


def encode_x(x, word2embedding, feature_header, max_seq_len,
             embed_dim, uncased):
    """Encode the input data.

    Arguments:
    x -- a Pandas dataframe
    word2embedding -- a dict(str -> np.array) from tokens to embeddings
    feature_header -- dataframe names of additional feature columns
    max_seq_len -- the maximum number of tokens per sentence in x
    embed_dim -- the array length of the vectors in word2embedding
    """
    embedding_matrix = np.zeros([len(x), max_seq_len,
                                 embed_dim + len(feature_header)])
    for row in x.itertuples():
        sent_idx = row.Index - 1
        for tok_idx in range(row.n_toks):
            word = str(row.token[tok_idx])
            if uncased:
                word = word.lower()
            embedding_matrix[sent_idx][tok_idx][:embed_dim] = \
                word2embedding.get(word, np.random.randn(embed_dim))
            for i, feature in enumerate(feature_header):
                embedding_matrix[sent_idx][tok_idx][embed_dim + i] = \
                    getattr(row, feature)[tok_idx]
    return embedding_matrix


def encode_y(y, label2idx, max_seq_len, n_classes):
    if n_classes == 1:
        labels = np.zeros([len(y), max_seq_len])
    else:
        labels = np.zeros([len(y), max_seq_len, n_classes])

    for row in y.itertuples():
        sent_idx = row.Index - 1
        for tok_idx, label in enumerate(row.label):
            labels[sent_idx][tok_idx] = label2idx[label]
    return labels


def prepare_data(config, word2embedding, training):
    # We're getting the comments this way so we can:
    # - add them to the output
    # - parse lines that actually contain '#' as token
    if training:
        infile = config.TRAIN_URL
    else:
        infile = config.DEV_URL
    comments = get_comments(infile, config.ONLINE_SOURCES)
    df = pd.read_csv(infile, sep='\t', skiprows=len(comments), quoting=3)

    std_cols = ['document_id', 'sent_id', 'token_start',
                'token_end', 'token', 'label']
    feature_cols = []
    for col in df.columns:
        if col not in std_cols:
            feature_cols.append(col)

    x_raw = get_features(df, feature_cols)
    x_enc = encode_x(x_raw, word2embedding, feature_cols,
                     config.MAX_SEQ_LEN, config.EMBED_DIM, config.UNCASED)

    y = None
    sample_weight = None
    if 'label' in df.columns:
        y_raw = get_cols(df, 'label')
        if config.N_CLASSES == 3:
            label2idx = {"O": [1, 0, 0], "B": [0, 0, 1], "I": [0, 1, 0]}
        elif config.N_CLASSES == 2:
            label2idx = {"O": [1, 0], "B": [0, 1], "I": [0, 1]}
        y = encode_y(y_raw, label2idx, config.MAX_SEQ_LEN, config.N_CLASSES)
        label2weight = {'O': config.O_WEIGHT, 'I': config.I_WEIGHT,
                        'B': config.B_WEIGHT}
        sample_weight = encode_y(y_raw, label2weight, config.MAX_SEQ_LEN,
                                 n_classes=1)

    return df, x_raw, x_enc, y, sample_weight, comments


def load_zipped_embeddings(infile):
    word2embedding = {}
    with zipfile.ZipFile(infile) as f_in_zip:
        file_in = f_in_zip.filelist[0].filename
        i = 0
        with f_in_zip.open(file_in, 'r') as f_in:
            for line in f_in:
                values = line.decode().rstrip().split()
                word2embedding[values[0]] = np.asarray(values[1:],
                                                       dtype='float32')
                i += 1
                if i % 100000 == 0:
                    print("Read " + str(i) + " embeddings")
    return word2embedding


def get_data(config, word2embedding=None):
    if not word2embedding:
        if config.EMBEDDING_PATH[-4:] == '.zip':
            word2embedding = load_zipped_embeddings(config.EMBEDDING_PATH)
        else:
            word2embedding = {}
            f = open(config.EMBEDDING_PATH)
            for line in f:
                values = line.rstrip().split()
                word2embedding[values[0]] = np.asarray(values[1:],
                                                       dtype='float32')
            f.close()

    _, _, train_x, train_y, sample_weight, comments = prepare_data(
        config, word2embedding, training=True)
    dev_df, dev_raw, dev_x, _, _, _ = prepare_data(config, word2embedding,
                                                   training=False)
    return Data(train_x, train_y, sample_weight, comments,
                dev_df, dev_raw, dev_x)


class Data:
    def __init__(self,
                 # If initializing on the fly:
                 train_x=None, train_y=None, sample_weight=None,
                 comments=None, dev_df=None, dev_raw=None, dev_x=None,
                 # If initializing from files:
                 path=None):
        self.train_x = train_x
        self.train_y = train_y
        self.sample_weight = sample_weight
        self.comments = comments
        self.dev_df = dev_df
        self.dev_raw = dev_raw
        self.dev_x = dev_x
        if path:
            self.load(path)


    def save(self, path='gdrive/My Drive/colab_projects/data/data/'):
        np.save(path + 'train_x', self.train_x)
        np.save(path + 'train_y', self.train_y)
        np.save(path + 'dev_x', self.dev_x)
        np.save(path + 'sample_weight', self.sample_weight)
        self.dev_raw.to_csv(path + 'dev_raw')
        self.dev_df.to_csv(path + 'dev_df')
        with open(path + 'comments.txt', 'w', encoding='utf8') as f:
            for comment in self.comments:
                f.write(comment + '\n')


    def load(self, path='gdrive/My Drive/colab_projects/data/data/'):
        self.train_x = np.load(path + 'train_x.npy')
        self.train_y = np.load(path + 'train_y.npy')
        self.dev_x = np.load(path + 'dev_x.npy')
        self.sample_weight = np.load(path + 'sample_weight.npy')
        self.dev_raw = pd.read_csv(path + 'dev_raw')
        self.dev_df = pd.read_csv(path + 'dev_df')
        self.comments = []
        with open(path + 'comments.txt', 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.comments.append(line)
        # self.print_summary()


    def print_summary(self):
        print('train_x')
        print(self.train_x.shape)
        print(self.train_x)
        print('\ntrain_y')
        print(self.train_y.shape)
        print(self.train_y)
        print('\ndev_x')
        print(self.dev_x.shape)
        print(self.dev_x)
        print('\nsample_weight')
        print(self.sample_weight.shape)
        print(self.sample_weight)
        print('\ndev_raw')
        print(self.dev_raw.info(verbose=True))
        print(self.dev_raw.head())
        print('\ndev_df')
        print(self.dev_df.info(verbose=True))
        print(self.dev_df.head())
        print('\ncomments')
        print(self.comments)


######################
# Creating the model #
######################


def get_bilstm(input_shape, config):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(config.LSTM_UNITS,
                                      return_sequences=True),
                            input_shape=input_shape))
    model.add(Dropout(config.DROPOUT))
    model.add(TimeDistributed(Dense(config.N_CLASSES, activation='softmax')))
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER,
                  metrics=[config.METRIC], sample_weight_mode='temporal')
    return model


def create_and_fit_bilstm(config, train_x, train_y, sample_weight):
    model = get_bilstm(train_x.shape[1:], config)
    history = model.fit(train_x, train_y, epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        sample_weight=sample_weight, verbose=1,)
    return model, history


###############
# Predictions #
###############


def get_bio_predictions(model, x, x_raw, n_classes, load_data):
    y_hat = model.predict(x)
    y_hat = y_hat.reshape(-1, n_classes).argmax(axis=1).reshape(x.shape[:2])
    labels = []
    for row in x_raw.itertuples():
        if load_data:
            sent_idx = row.sent_id - 1
        else:
            sent_idx = row.Index - 1
        for tok_idx in range(row.n_toks):
            if y_hat[sent_idx][tok_idx] == 0:
                label = "O"
            elif y_hat[sent_idx][tok_idx] == 1:
                label = "I"
            else:
                label = "B"
            labels.append(label)
    return labels


def si_predictions_to_spans(label_df):
    spans = []
    prev_label = 'O'
    prev_span_start = '-1'
    prev_span_end = '-1'
    prev_article = ''

    for row in label_df.itertuples():
        article = row.document_id
        span_start = row.token_start
        span_end = row.token_end
        label = row.label

        span, prev_span_start = update_prediction(article, label,
                                                  span_start, span_end,
                                                  prev_article, prev_label,
                                                  prev_span_start,
                                                  prev_span_end)
        if span is not None:
            spans.append(span)

        prev_article = article
        prev_label = label
        prev_span_end = span_end

    # Make sure we get the last prediction
    span, _ = update_prediction(article, label, span_start, span_end,
                                prev_article, prev_label, prev_span_start,
                                prev_span_end)
    if span is not None:
        spans.append(span)
    return spans


# Helper method for si_predictions_to_spans
def update_prediction(article, label, span_start, span_end, prev_article,
                      prev_label, prev_span_start, prev_span_end):
    span = None
    cur_span_start = prev_span_start
    # Ending a span: I-O, B-O, I-B, B-B, new article
    if prev_label != 'O' and (label != 'I' or prev_article != article):
        span = (prev_article, prev_span_start, prev_span_end)

    # Starting a new span: O-B, O-I, I-B, B-B, new article
    if label == 'B' or (label == 'I' and prev_label == 'O') \
            or prev_article != article:
        # Update the start of the current label span
        cur_span_start = span_start
    return span, cur_span_start


def print_spans(spans, file_prefix, file_stem, file_suffix):
    outfile = file_prefix + 'spans_' + file_stem + '_' + file_suffix + '.txt'
    with open(outfile, mode='w') as f:
        for span in spans:
            f.write(str(span[0]) + '\t' + str(span[1]) + '\t' +
                    str(span[2]) + '\n')


def predict(config, model, history, dev_df, dev_raw, dev_x, comments,
            file_prefix, file_stem, file_suffix, predict_spans=True):
    y_hat = get_bio_predictions(model, dev_x, dev_raw, config.N_CLASSES, config.LOAD_DATA)
    result_df = pd.concat([dev_df, pd.DataFrame(y_hat, columns=['label'])],
                          axis=1, sort=False)

    logfile = file_prefix + 'log_' + file_stem + '_' + file_suffix + '.txt'

    with open(logfile, mode='w') as f:
        f.write('DATA PREPROCESSING\n\n')
        for comment in comments:
            comment = comment.replace('#', '')
            fields = comment.split(',')
            for field in fields:
                f.write(comment.strip() + '\n')
        f.write('\n\nCONFIG\n\n')
        f.write(config.pretty_str())
        f.write('\n\nMODEL HISTORY\n\n')
        f.write('Loss ' + config.LOSS + '\n')
        f.write(str(history.history['loss']) + '\n')
        f.write(config.METRIC + '\n')
        f.write(str(history.history[config.METRIC]) + '\n')
        f.write('\n\nMODEL SUMMARY\n\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    if predict_spans:
        spans = si_predictions_to_spans(result_df)
        print_spans(spans, file_prefix, file_stem, file_suffix)

    return result_df


###########################
# Putting it all together #
###########################


def run(config, file_stem, file_suffix, verbose=True, predict_spans=True,
        data=None, word2embedding=None, file_prefix=''):
    if verbose:
        print('Running with config:')
        print(config.pretty_str())
    if not data:
        if config.LOAD_DATA:
            print('Loading data from files')
            data = Data(path=config.DATA_PATH)
        else:
            if verbose:
                print('Encoding the data')
            data = get_data(config, word2embedding)
            if config.SAVE_DATA:
                data.save()
    if verbose:
        print('Building the model')
    model, history = create_and_fit_bilstm(config, data.train_x,
                                           data.train_y,
                                           data.sample_weight)
    if verbose:
        print('Predicting the test data spans')
    labels = predict(config, model, history, data.dev_df, data.dev_raw,
                     data.dev_x, data.comments, file_prefix, file_stem,
                     file_suffix, predict_spans)
    if verbose:
        print('Done!\n\n')

    return data, labels