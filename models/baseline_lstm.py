from keras.models import Sequential
from keras.layers import *
from utils.prep_data_for_model import make_x_and_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

EMBEDDING_SIZE = 100
#max length of sentence we care about
MAX_SEQUENCE_LENGTH = 30

def set_up_inputs_outputs():
    x,y = make_x_and_y()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print("test set size " + str(len(X_test)))
    return X_train, X_test, y_train, y_test

def build_model():
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE)))
    # model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_sequence_length, embedding_dim)))
    model.add(Dropout(0.25))

    # model.add(Dense(units=5, activation='softmax'))
    model.add(TimeDistributed(Dense(3, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def compute_f1(y_test, y_hat):
    print(f1_score(y_test, y_hat, average="macro"))


def one_to_idx(input_matrix):
    out = []
    for sent in input_matrix:
        sent_rep = []
        out.append(np.argmax(sent, axis=1))
    return out

if __name__ == '__main__':
    batch_size = 128

    X_train, X_test, y_train, y_test = set_up_inputs_outputs()
    model = build_model()
    history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=1, validation_split=0.1)
    y_hat = model.predict(X_test)

    compute_f1(y_test, y_hat)

    print(one_to_idx(y_test)[0])

    print(y_test[0])