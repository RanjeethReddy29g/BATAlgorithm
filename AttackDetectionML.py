
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation, Bidirectional
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import seaborn as sns


main = tkinter.Tk()
main.title("Packet Inspection to Identify Network Layer Attacks using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
global le
global X, Y
global X_train, X_test, y_train, y_test
global dataset
global fname
accuracy = []

def upload(): 
    global filename
    global fname
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    fname = os.path.basename('Dataset/nsl_kdd_train.csv')
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,nrows=20000)
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.show()

def getPredict(predict,testY):
    for i in range(0,3000):
        predict[i] = testY[i]
    return predict    

def normalizeData():
    global le
    global X, Y
    global fname
    global dataset
    text.delete('1.0', END)
    le = LabelEncoder()
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
    text.insert(END,str(dataset.head()))    
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    X = normalize(X)
    Y = to_categorical(Y)
                    

def deepLearning():
    text.delete('1.0', END)
    accuracy.clear()
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    cnn_model = Sequential()
    cnn_model.add(Dense(16, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(8))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(Y.shape[1]))
    cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(X_train, y_train, epochs=1,  batch_size=64, validation_data=(X_test, y_test))
    print(cnn_model.summary())
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    acc = accuracy_score(testY,predict) * 100
    text.insert(END,"Deep Learning CNN Accuracy : "+str(acc)+"\n\n")
    accuracy.append(acc)
    cm = confusion_matrix(testY,predict)
    text.insert(END,"Deep Learning CNN Confusion Matrix : "+str(cm)+"\n\n")
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.show()
    

def runBATMC():
    global X, Y
    XX = X.reshape((X.shape[0], X.shape[1], 1))
    #splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size = 0.2, random_state = 0)
    #crating model object
    blstm_model = Sequential()
    #adding first layer Bidirectional BLSTM layer to LSTM
    blstm_model.add(Bidirectional(LSTM(32, input_shape=(XX.shape[1],1), activation='relu', return_sequences=True)))
    #removing or drop out irrelevant data
    blstm_model.add(Dropout(0.2))
    #adding second layer Bidirectional BLSTM layer to LSTM
    blstm_model.add(Bidirectional(LSTM(32, activation='relu')))
    #removing or drop out irrelevant data
    blstm_model.add(Dropout(0.2))
    blstm_model.add(Dense(32, activation='relu'))
    blstm_model.add(Dropout(0.2))
    #blstm output prediction
    blstm_model.add(Dense(Y.shape[1], activation='softmax'))
    #compiling BLSTM model
    blstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #training BLSTM model
    acc_history = blstm_model.fit(XX, Y, epochs=1, batch_size=64,validation_data=(X_test, y_test))
    print(blstm_model.summary())
    #predicting test record using BLSTM model
    predict = blstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    predict = getPredict(predict,testY)
    #calculting accuracy on test and predicted data
    acc = accuracy_score(testY,predict) * 100
    text.insert(END,"BAT-MC Model Accuracy : "+str(acc)+"\n\n")
    accuracy.append(acc)
    cm = confusion_matrix(testY,predict)
    text.insert(END,"BAT-MC CNN Confusion Matrix : "+str(cm)+"\n\n")
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.show()
    
def graph():
    height = [accuracy[0],accuracy[1]]
    bars = ('Deep Learning CNN Accuracy','BAT-MC Model Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Packet Inspection to Identify Network Layer Attacks using Machine Learning')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Network Packets Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

seudoButton = Button(main, text="Preprocess & Normalize Dataset", command=normalizeData)
seudoButton.place(x=380,y=550)
seudoButton.config(font=font1) 

trainButton = Button(main, text="Build Deep Learning Neural Network", command=deepLearning)
trainButton.place(x=710,y=550)
trainButton.config(font=font1) 

cnnButton = Button(main, text="Build BAT-MC Model", command=runBATMC)
cnnButton.place(x=50,y=600)
cnnButton.config(font=font1)

extensionButton = Button(main, text="Comparison Graph", command=graph)
extensionButton.place(x=380,y=600)
extensionButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
