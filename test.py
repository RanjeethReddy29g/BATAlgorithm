import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation, Bidirectional
from sklearn.preprocessing import normalize

le = LabelEncoder()

dataset = pd.read_csv('Dataset/nsl_kdd_train.csv',nrows=20000)
fname = os.path.basename('Dataset/nsl_kdd_train.csv')
label = dataset.groupby('label').size()
label.plot(kind="bar")
plt.show()
if fname == 'kddcup.csv':
    dataset['protocol_type'] = pd.Series(le.fit_transform(dataset['protocol_type'].astype(str)))
    dataset['service'] = pd.Series(le.fit_transform(dataset['service'].astype(str)))
    dataset['flag'] = pd.Series(le.fit_transform(dataset['flag'].astype(str)))
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
if fname == 'nsl_kdd_train.csv':
    dataset['protocol_type'] = pd.Series(le.fit_transform(dataset['protocol_type'].astype(str)))
    dataset['service'] = pd.Series(le.fit_transform(dataset['service'].astype(str)))
    dataset['flag'] = pd.Series(le.fit_transform(dataset['flag'].astype(str)))
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))


dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
print(X.shape)
X = normalize(X)
Y = to_categorical(Y)
#X = X.reshape((X.shape[0], X.shape[1], 1))
print(X)
print(Y)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(Y.shape)
'''
lstm_model = Sequential()
lstm_model.add(Bidirectional(LSTM(128, input_shape=(X.shape[1],1), activation='relu', return_sequences=True)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Bidirectional(LSTM(128, activation='relu')))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(Y.shape[1], activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
acc_history = lstm_model.fit(X, Y, epochs=1, batch_size=64,validation_data=(X_test, y_test))
print(lstm_model.summary())
predict = lstm_model.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
acc = accuracy_score(testY,predict) * 100
print(acc)
'''

cnn_model = Sequential()
cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(512))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(Y.shape[1]))
cnn_model.add(Activation('softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
acc_history = cnn_model.fit(X, Y, epochs=1,  batch_size=64, validation_data=(X_test, y_test))
print(cnn_model.summary())
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
acc = accuracy_score(testY,predict) * 100
print(acc)







