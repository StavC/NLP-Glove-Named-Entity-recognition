
from sklearn import svm
from sklearn.metrics import *
import pickle

from HandleData import TestDataToSen, TestWordsToGloveRep


def TrainSVM(train_x,train_y): #training the SVM model and saving his weights
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
    for sen in test_full_vecs:#predicting each word and appending it to a list of predictions with values T and O
        for word in sen:
            pred_svm = svm_model.predict(word.reshape(1, -1))
            if pred_svm == True:
                pred_y.append('T')
            else:
                pred_y.append('O')
        pred_y.append('')
    return sentences, pred_y