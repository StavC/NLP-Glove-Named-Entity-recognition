from gensim import downloader
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import re
from sklearn import svm
from sklearn.metrics import *
import pickle


def TaggedDataToSen(text,name):  # Reading all words in the file, adding them to lists per sentence with some padding
    with open(text, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.split() for sen in sentences if sen]

    countWords = 0

    current_sen = []
    current_sen.append('unkunkunk')
    current_sen.append('unkunkunk')
    current_sen.append('unkunkunk')
    current_labels = []
    all_sen_labels = []
    all_sen = []
    for word in sentences:
        if word == []:  # if we reached the end of the sen
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            all_sen.append(current_sen)  # append the sen to the list of all sen
            all_sen_labels.append(current_labels)
            current_sen = []  # start a new sen
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_labels = []
            continue

        current_sen.append(word[0])
        countWords += 1

        if word[1] == 'O':  # change the labels to binary tagging
            current_labels.append(False)
        else:
            current_labels.append(True)
    print(f' there are {countWords} words and labels in {name} dataset')

    return all_sen, all_sen_labels


def TestDataToSen(text):  # Reading all words in the file, adding them to lists per sentence with some padding
    with open(text, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.split() for sen in sentences if sen]

    countWords = 0

    current_sen = []
    current_sen.append('unkunkunk')
    current_sen.append('unkunkunk')
    current_sen.append('unkunkunk')
    current_labels = []
    all_sen_labels = []
    all_sen = []
    for word in sentences:
        if word == []:  # if we reached the end of the sen
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            all_sen.append(current_sen)  # append the sen to the list of all sen
            all_sen_labels.append(current_labels)
            current_sen = []  # start a new sen
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_sen.append('unkunkunk')
            current_labels = []
            continue

        current_sen.append(word[0])
        countWords += 1

    print(f' there are {countWords} words in Test dataset')

    return all_sen


def WordToGloveRep(sens, labels, glove, name):
    vecWords = []
    vecLabels = []

    UNKcount = 0
    inVocab = 0
    outVocab = 0

    for i, sen in enumerate(sens):  # for each sen

        for j, word in enumerate(sen):  # going through each word
            word = word.lower()
            if word == 'unkunkunk':  # if we are inspecting a "pad" token we will init it as a vector of zeros
                sens[i][j] = np.zeros((50,))
                UNKcount += 1
            elif word in glove.key_to_index:  # if word is in our vocab then assign its vector
                sens[i][j] = glove[word]
                inVocab += 1
            elif word not in glove.key_to_index:  # if the word is oov we will try to remove some chars from it and check if it's now a part of our vocab
                newword = re.sub("[.,!?:;$#-='...\"@#_]", "", word)
                if newword in glove.key_to_index:
                    sens[i][j] = glove[newword]
                    inVocab += 1
                else:  # if the word is not in our vocab then assign a vector to it
                    sens[i][j] = np.ones((50,)) * -0.1
                    outVocab += 1

    print(f'Stats for the {name} set:  UNK: {UNKcount}  In Vocab {inVocab} Out VOcab: {outVocab} ')

    for i, sen in enumerate(sens):  # for each sen

        for j, word in enumerate(sen):  # going through each word and representing it as a concatenated vector of nearby words

            if not (word.all() == np.zeros((50,)).all()): # if the word is not a "Pad" Token
                before = np.concatenate((sens[i][j - 3], sens[i][j - 2]))
                before2 = np.concatenate((before, sens[i][j - 1]))

                after = np.concatenate((sens[i][j + 1], sens[i][j + 2]))
                after2 = np.concatenate((after, sens[i][j + 3]))

                mid = np.concatenate((before2, word))
                mid2 = np.concatenate((mid, after2))
                vecWords.append(mid2)
                vecLabels.append(labels[i][j - 3])


    return vecWords, vecLabels

def TrainSVM(train_x,train_y):
    svm_model = svm.SVC()
    svm_model.fit(train_x, train_y)
    with open('NewSVMModel.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    return svm_model

def LoadSVMModel():
    with open('SVMModel.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model

def ValPredictSVM(val_x,val_y,svm_model):
    val_y_pred_svm = svm_model.predict(val_x)
    print(f'F1 score for the val {f1_score(val_y, val_y_pred_svm)}')
    print(f'Accuracy score for the val {accuracy_score(val_y, val_y_pred_svm)}')
    print(classification_report(val_y, val_y_pred_svm))

def TrainPredictSVM(train_x,train_y,svm_model):
    train_y_pred_svm = svm_model.predict(train_x)
    print(f'F1 score for the train {f1_score(train_y, train_y_pred_svm)}')
    print(f'Accuracy score for the train {accuracy_score(train_y, train_y_pred_svm)}')
    print(classification_report(train_y, train_y_pred_svm))

def TestPredictSVM(test_x,svm_model):
    train_y_pred_svm = svm_model.predict(test_x)



def main():
    # Getting started
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    train_path = 'train.tagged'
    dev_path = 'dev.tagged'
    test_path='test.untagged'
    train_full_sen, train_full_labels = TaggedDataToSen('train.tagged',name='Train')
    val_full_sen, val_full_labels = TaggedDataToSen('dev.tagged',name='Validation')
    train_x, train_y = WordToGloveRep(train_full_sen, train_full_labels, glove, name='Train')
    val_x, val_y = WordToGloveRep(val_full_sen, val_full_labels, glove, name='Validation')

    #training part
    TrainSVM(train_x, train_y)
    # loading the SVM Model
    svm_model=LoadSVMModel()
    TrainPredictSVM(train_x, train_y, svm_model)
    ValPredictSVM(val_x, val_y, svm_model)

if __name__ == "__main__":
    main()
