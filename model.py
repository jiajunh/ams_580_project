from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from utils import compute_scores


def train_logistic_model(X_train, Y_train, args):
    model = LogisticRegression(random_state=args.seed,
                               max_iter=500,
                               penalty="l2",
                            #    C=1.0, 
                            #    l1_ratio=0.5,
                            #    solver="saga",
                               ).fit(X_train, Y_train)
    return model


def train_xgboost(X_train, Y_train, args):
    model = XGBClassifier(n_estimators=200,
                          max_depth=4, 
                          learning_rate=0.1, 
                          min_child_weight=2)
    model.fit(X_train, Y_train)
    return model



def train_svm_model(X_train, Y_train, args):
    model = SVC(C=1.0,
                gamma="auto",
                kernel="rbf",
                random_state=args.seed)
    model.fit(X_train, Y_train)
    return model

def train_random_forest(X_train, Y_train, args):
    model = RandomForestClassifier(n_estimators=200,
                                   criterion="gini",
                                   max_depth=8, 
                                   random_state=args.seed)
    model = model.fit(X_train, Y_train)
    return model



class Net(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(Net, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.dropout1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.fc3(out))
        return out

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.labels[idx]
        return sample

def eval_nn_model(model, X_val, Y_val):
    val_dataset = CustomDataset(X_val, Y_val)
    dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model.eval()
    pred = torch.empty((0, 1), dtype=torch.float32)
    y = torch.empty((0), dtype=torch.float32)
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            pred = torch.cat((pred, outputs), dim=0)
            y = torch.cat((y, labels), dim=0)
        pred = pred.numpy().reshape(y.shape)
        y = y.numpy().astype(int)
        pred = (pred > 0.5).astype(int)
    return y, pred


def train_neural_network(X_train, Y_train, X_val, Y_val, args):
    batch_size = 256
    epoch = 5
    learning_rate = 0.015
    weight_decay = 0.0

    model = Net(X_train.shape[1])
    dataset = CustomDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epoch, gamma=0.8)

    best_model = Net(X_train.shape[1]).eval()
    best_score = 0.0

    for i in range(epoch):
        print("-"*20, f"Epoch {i+1}/{epoch}", "-"*20)
        model.train()
        for inputs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs).reshape(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        y, pred = eval_nn_model(model, X_val, Y_val)
        val_scores = compute_scores(y, pred)
        if val_scores[-1] > best_score:
            best_model.load_state_dict(model.state_dict())
            best_score = val_scores[-1]
        print(f"Epoch [{i+1}/{epoch}], Loss: {loss.item():.4f}")
    
    return best_model