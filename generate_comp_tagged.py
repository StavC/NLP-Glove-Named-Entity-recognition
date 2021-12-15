
from ANN import *
from SVM import *
from HandleData import *

def generate_comp_tagged():
    print('Working on inference.....')
    GLOVE_PATH = 'glove-twitter-50'
    glove = downloader.load(GLOVE_PATH)
    test_path='test.untagged'

    ########SVM Part########
    svm_model = LoadSVMModel('SVMModel.pkl')  # loading the SVM Model
    sentences, pred_y = TestPredictSVM(test_path, svm_model, glove)
    WriteTestTaggedSVM(sentences, pred_y)


    #########ANN Part########
    ANN_model = LoadANNModel('saved_weights.pt')  # if you want to train the model you can load it with the name NEW_saved_weights.pt
    sentences, pred_y = TestPredictANN(test_path, ANN_model, glove)
    WriteTestTaggedANN(sentences, pred_y)


if __name__ == "__main__":
    generate_comp_tagged()
