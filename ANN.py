import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import *
from HandleData import *
from torch.autograd import Variable

torch.manual_seed(0)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(350, 2000)
        self.layer_extra = torch.nn.Linear(2000, 1000)
        self.layer_extra2 = torch.nn.Linear(1000, 500)
        self.layer_extra3 = torch.nn.Linear(500, 250)
        self.layer_2 = torch.nn.Linear(250, 1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_extra(x))
        x = F.relu(self.layer_extra2(x))
        x = F.relu(self.layer_extra3(x))
        x = self.layer_2(x)
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def trainANN(train_x,train_y, val_x,val_y):
    model = ANN()
    print(model)
    # Define Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.BCEWithLogitsLoss()
    batch_size = 512

    # create Tensor datasets
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    train_x = np.array(train_x)
    val_x = np.array(val_x)

    train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    valid_data = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    #  checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = model.float()
    model.to(device)
    model.train()

    epochs = 5
    min_valid_loss = np.inf

    for e in range(1, epochs + 1):

        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = loss_func(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        valid_loss = 0.0
        valid_acc = 0.0
        model.eval()
        for data, labels in valid_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)

            # Find the Loss and Acc
            loss = loss_func(target, labels.unsqueeze(1))
            acc = binary_acc(target, labels.unsqueeze(1))

            # Calculate Loss and Acc
            valid_loss += loss.item()
            valid_acc += acc.item()

        if valid_loss < min_valid_loss: #saving the best model according to val loss
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), 'NEW_saved_weights.pt')

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}, |  Validation Loss: { valid_loss / len(valid_loader)} | Validation Acc: {valid_acc / len(valid_loader)}')

def LoadANNModel(path):
    with torch.no_grad():
        model = ANN()
        model.load_state_dict(torch.load(path))
        print(model)
    return model
def PredictF1onVal(model,val_x,val_y):
    val_y = np.array(val_y)
    val_x = np.array(val_x)
    y_pred_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        print(model)
        model.to(device)
        valid_data = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=512)

        for data, labels in valid_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)
            target = target.to('cpu')
            y_test_pred = torch.sigmoid(target)
            y_pred_tag = torch.round(y_test_pred)

            for tag in y_pred_tag:
                y_pred_list.append(tag[0])

    print(f' normal F1 from Sklearn {f1_score(val_y, y_pred_list)}')
    print(classification_report(val_y, y_pred_list))
    print('Confusion Matrix')
    print(confusion_matrix(val_y, y_pred_list))




def TestPredictANN(test_path, ANN_model, glove):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ANN_model.to(device)
    ANN_model.eval()
    test_full_sen = TestDataToSen(test_path)
    test_full_vecs = TestWordsToGloveRep(test_full_sen, glove)
    test_full_vecs = np.array(test_full_vecs,dtype=object)
    pred_y = []


    with open(test_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.split() for sen in sentences if sen]

    for sen in test_full_vecs:
        for word in sen:
            word=np.array(word)
            row = Variable((torch.Tensor(word).float()).to(device))
            pred_ANN = ANN_model(row.reshape(1, -1))
            y_test_pred = torch.sigmoid(pred_ANN)
            y_pred_tag = torch.round(y_test_pred)
            if y_pred_tag == True:
                pred_y.append('T')
            else:
                pred_y.append('O')
        pred_y.append('')
    return sentences, pred_y