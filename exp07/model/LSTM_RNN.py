from keras.datasets import imdb
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras_preprocessing import sequence
from matplotlib import pyplot as plt


def lstm():
    max_features = 10000
    maxlen = 500
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="/home/raymond/project/Experiment-Sci/exp07/data/imdb.npz", num_words=max_features)
    print(len(X_train), "train sequences")
    print(len(X_train), "test sequences")
    print("Pad sequences (sample x times)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    # Plot training history
    plot_history(history)

    model.summary()


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('../result/LSTM_RNN_accuracy.png')
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('../result/LSTM_RNN_loss.png')
    plt.show()


if __name__ == "__main__":
    lstm()
