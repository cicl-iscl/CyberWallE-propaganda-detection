# -*- coding: utf-8 -*-
"""BaselineLSTM.ipynb

"""
"""Inspiration for glove embeddings: https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/"""

import pandas as pd
import numpy as np

EMBEDDINGS_FILE = '../tools/embeddings/glove.6B.100d.txt'
EMBEDDING_SIZE = 100
DATA_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/train-data-bio.tsv?token=AFDEFD7WXGQPGEJK6X5OB6C53AMEC'
# max length of sentence we care about
MAX_SEQUENCE_LENGTH = 30
N_CLASSES = 3


def read_in_data(data_url):
    full_dataframe = pd.read_csv(data_url, sep='\t',
                                 names=["document_id", "sent_number", "idx_token_beginning", "idx_token_end", "token",
                                        "bio_label"], quoting=3)
    print(full_dataframe.head())
    return full_dataframe


def isolate_token_list(full_dataframe):
    df2 = full_dataframe.token.astype(str)
    df3 = df2.str.lower()
    token_list = df3.tolist()
    print(token_list[0:5])
    tokens_set = set(df3)
    print("There are " + str(len(token_list)) + " tokens in the dataset overall.")
    print("There are " + str(len(tokens_set)) + " unique tokens in the dataset.")
    return token_list


"""Group by sentence number - extract sentences from dataframe"""


def isolate_sent_list(full_dataframe):
    df_sents = full_dataframe.groupby('sent_number')['token'].apply(list)
    print(df_sents.head())
    df_sents = df_sents.to_frame()
    print(list(df_sents.columns.values))
    return df_sents


def isolate_labels_list(full_dataframe):
    df_labels = full_dataframe.groupby('sent_number')['bio_label'].apply(list)
    df_labels = df_labels.to_frame()
    df_labels.head()
    bio_sent_list = df_labels["bio_label"].to_list()
    return bio_sent_list


def isolate_sents_tokenized(df_sents, inpect_sents=True):
    df_sents['sent_number'] = df_sents.index
    df_sents["sentences"] = df_sents["token"].str.join(" ")
    df_sents.head()
    # print(list(df_sents.columns.values))
    if inpect_sents:
        inspect_sentences(df_sents)
    tokenized_sent_lists = df_sents["token"].to_list()
    return tokenized_sent_lists


def inspect_sentences(df_sents):
    df_sents['len'] = df_sents['token'].apply(lambda x: len(x))
    print("mean length of sentence: " + str(df_sents.len.mean()))
    print("max length of sentence: " + str(df_sents.len.max()))
    print("std dev length of sentence: " + str(df_sents.len.std()))


def generate_expected_outputs(n_classes, bio_sent_list, tokenized_sent_lists):
    """Generates expected one hot BIO labels for tokens in each sentence. One hots in an innovative new way!!!"""
    n_classes = 3
    label2idx = {"O": [1, 0, 0], "B": [0, 1, 0], "I": [0, 0, 1]}
    # first create a matrix of zeros, this is our embedding matrix
    y = np.zeros([len(tokenized_sent_lists), MAX_SEQUENCE_LENGTH, n_classes])
    # for each word in out tokenizer lets try to find that work in our w2v model
    for i, sentence in enumerate(tokenized_sent_lists):
        for j, word in enumerate(bio_sent_list[i]):
            word = normalize_token(word)
            if j < MAX_SEQUENCE_LENGTH:
                y[i][j] = label2idx.get(word)
            else:
                break
    return y


def get_pretrained_embeddings(embeddings_file):
    embeddings_index = {}

    f = open(embeddings_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def normalize_token(token):
    str(token).lower()
    return token


def generate_embeddings_as_input(embeddings_index, tokenized_sent_lists):
    embedding_dim = 100
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros([len(tokenized_sent_lists), MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE])
    # for each word in out tokenizer lets try to find that work in our w2v model
    for i, sentence in enumerate(tokenized_sent_lists):
        for j, word in enumerate(tokenized_sent_lists[i]):
            if j > MAX_SEQUENCE_LENGTH:
                # Split these longer sentences later
                break
            word = normalize_token(word)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # we found the word - add that words vector to the matrix
                embedding_matrix[i] = embedding_vector
            else:
                # doesn't exist, assign a random vector
                embedding_matrix[i] = np.random.randn(embedding_dim)

    print('Embedding shape: ', embedding_matrix.shape)

    return embedding_matrix


def make_x_and_y():
    full_dataframe = read_in_data(DATA_URL)
    df_sents = isolate_sent_list(full_dataframe)
    # token_list = isolate_token_list(full_dataframe)
    bio_sent_list = isolate_labels_list(full_dataframe)
    tokenized_sent_lists = isolate_sents_tokenized(df_sents)
    y = generate_expected_outputs(N_CLASSES, bio_sent_list, tokenized_sent_lists)
    embeddings_index = get_pretrained_embeddings(EMBEDDINGS_FILE)
    x = generate_embeddings_as_input(embeddings_index, tokenized_sent_lists)
    return x, y
