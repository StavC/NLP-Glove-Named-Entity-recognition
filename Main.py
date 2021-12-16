from gensim import downloader
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
    train_x,train_y=BalanceTrainData(train_x,train_y) # Balancing the TrainData with oversampling and undersampling
    print('Finished loading train, val and balance train, moving on to the SVM part')

    #########SVM Part########
    svm_model=TrainSVM(train_x, train_y) # Train SVM and saving a model named :NewSVMModel.pkl
    svm_model=LoadSVMModel('NewSVMModel.pkl')   # loading the SVM Model
    TrainPredictSVM(train_x, train_y, svm_model) #making prediction on train data
    ValPredictSVM(val_x, val_y, svm_model)
    sentences, pred_y = TestPredictSVM(test_path, svm_model, glove) # making predictions on test data
    WriteTestTaggedSVM(sentences, pred_y) # writing a file called SVMtest.tagged
    print('Finished the SVM part, moving on to the ANN part')


    #########ANN Part########
    trainANN(train_x,train_y, val_x,val_y) # training the ANN and saving the model weights to NEW_saved_weights.pt
    ANN_model=LoadANNModel('NEW_saved_weights.pt')
    PredictF1onVal(ANN_model,val_x,val_y) # Predicting the F1 on the validation set
    sentences, pred_y = TestPredictANN(test_path, ANN_model, glove)  # making predictions on test data
    WriteTestTaggedANN(sentences, pred_y)  # writing a file called ANNtest.tagged
    print('Done :)')


if __name__ == "__main__":
    main()
