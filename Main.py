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
from sklearn.utils import shuffle


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


def TestWordsToGloveRep(sens, glove):
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

    print(f'Stats for the Test set:  UNK: {UNKcount}  In Vocab {inVocab} Out VOcab: {outVocab} ')
    vecSen = []
    for i, sen in enumerate(sens):  # for each sen
        vecWords = []
        for j, word in enumerate(sen):  # going through each word and representing it as a concatenated vector of nearby words

            if not (word.all() == np.zeros((50,)).all()):  # if the word is not a "Pad" Token
                before = np.concatenate((sens[i][j - 3], sens[i][j - 2]))
                before2 = np.concatenate((before, sens[i][j - 1]))

                after = np.concatenate((sens[i][j + 1], sens[i][j + 2]))
                after2 = np.concatenate((after, sens[i][j + 3]))

                mid = np.concatenate((before2, word))
                mid2 = np.concatenate((mid, after2))
                vecWords.append(mid2)
        vecSen.append(vecWords)

    return vecSen

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
def BalanceTrainData(train_x,train_y):
    """

    :param train_x:
    :param train_y:
    :return: a more balanced dataset by oversampling True labels and undersampling False labels, the returned dataset has a ratio of 1:5 T:F.
    """
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    train_x_over_sampled, train_y_over_sampled = train_x.copy(), train_y.copy()
    for word, label in zip(train_x, train_y): # oversampling the True labels 2X
        if label == True:
            train_x_over_sampled.append(word)
            train_y_over_sampled.append(label)
    train_x, train_y = train_x_over_sampled, train_y_over_sampled
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    trainX = []
    trainY = []
    countT = 0
    countF = 0
    for word, label in zip(train_x, train_y): # undersampling to a ratio of 5:1 F:T
        if label == True:
            trainX.append(word)
            trainY.append(label)
            countT += 1
        elif label == False and countF <= 2462 * 10:  # 5 was good
            trainX.append(word)
            trainY.append(label)
            countF += 1
    train_x = trainX
    train_y = trainY

    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    return train_x,train_y

def TrainSVM(train_x,train_y):
    svm_model = svm.SVC()
    svm_model.fit(train_x, train_y)
    with open('NewSVMModel.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    return svm_model

def LoadSVMModel(path):
    with open(path, 'rb') as f:
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


def TestPredictSVM(test_path, svm_model, glove):
    test_full_sen = TestDataToSen(test_path)
    test_full_vecs = TestWordsToGloveRep(test_full_sen, glove)

    with open(test_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.split() for sen in sentences if sen]

    pred_y = []
    for sen in test_full_vecs:
        for word in sen:
            pred_svm = svm_model.predict(word.reshape(1, -1))
            if pred_svm == True:
                pred_y.append('T')
            else:
                pred_y.append('O')
        pred_y.append('')
    return sentences, pred_y

def WriteTestTagged(sentences,pred_y):
    with open('test.tagged', 'w', encoding="utf-8") as f:
        for word, label in zip(sentences, pred_y):
            if word != []:
                f.write(str(word[0]) + '\t' + str(label) + '\n')
            else:
                f.write('\n')
    print('finished writing the file "test.tagged"')
def main():
    # Getting started
    GLOVE_PATH = 'glove-twitter-50'
    glove = downloader.load(GLOVE_PATH)
    train_path = 'train.tagged'
    dev_path = 'dev.tagged'
    test_path='test.untagged'
    train_full_sen, train_full_labels = TaggedDataToSen('train.tagged',name='Train')
    val_full_sen, val_full_labels = TaggedDataToSen('dev.tagged',name='Validation')
    train_x, train_y = WordToGloveRep(train_full_sen, train_full_labels, glove, name='Train')
    val_x, val_y = WordToGloveRep(val_full_sen, val_full_labels, glove, name='Validation')



    #training part
    train_x,train_y=BalanceTrainData(train_x,train_y)
    #svm_model=TrainSVM(train_x, train_y)

    svm_model=LoadSVMModel('SVMModel.pkl')   # loading the SVM Model

    TrainPredictSVM(train_x, train_y, svm_model)
    ValPredictSVM(val_x, val_y, svm_model)
    sentences, pred_y = TestPredictSVM(test_path, svm_model, glove)
    WriteTestTagged(sentences, pred_y)


if __name__ == "__main__":
    main()
