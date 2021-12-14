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
from ANN import *
from SVM import *
from HandleData import *




def main():
    # Getting started
    GLOVE_PATH = 'glove-twitter-50'
    glove = downloader.load(GLOVE_PATH)
    train_path = 'train.tagged'
    dev_path = 'dev.tagged'
    test_path='test.untagged'

    #########Handling Data Part########
    train_full_sen, train_full_labels = TaggedDataToSen(train_path,name='Train')
    val_full_sen, val_full_labels = TaggedDataToSen(dev_path,name='Validation')
    train_x, train_y = WordToGloveRep(train_full_sen, train_full_labels, glove, name='Train')
    val_x, val_y = WordToGloveRep(val_full_sen, val_full_labels, glove, name='Validation')
    train_x,train_y=BalanceTrainData(train_x,train_y)

    #########SVM Part########
    #svm_model=TrainSVM(train_x, train_y)
    svm_model=LoadSVMModel('NewSVMModel.pkl')   # loading the SVM Model
    #TrainPredictSVM(train_x, train_y, svm_model)
    #ValPredictSVM(val_x, val_y, svm_model)
    sentences, pred_y = TestPredictSVM(test_path, svm_model, glove)
    WriteTestTaggedSVM(sentences, pred_y)


    #########ANN Part########
    trainANN(train_x,train_y, val_x,val_y)
    ANN_model=LoadANNModel('saved_weights.pt') # if you want to train the model you can load it with the name NEW_saved_weights.pt
    PredictF1onVal(ANN_model,val_x,val_y)
    sentences, pred_y = TestPredictANN(test_path, svm_model, glove)
    WriteTestTaggedANN(sentences, pred_y)


if __name__ == "__main__":
    main()
