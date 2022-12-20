from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from copy import deepcopy

class EarlyStopper:
    def __init__(self, model, num_trials):
        self.num_trials = num_trials
        self.model = model
        self.trial_counter = 0
        self.best_metric = -1e9
        self.best_state = deepcopy(model.state_dict())

    def iscontinuable(self, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_state = deepcopy(self.model.state_dict())
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size=None, task='classification', device='cpu'):
        assert task in ['classification', 'regression']
        self.model = model
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.task = task
        self.device = device

    def train(self, train_x, train_y, name, epoch=100, trials=None, valid_x=None, valid_y=None):
        self.model.to(self.device)
        if self.batch_size:
            train_data = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        else:
            train_loader = [[train_x, train_y]]

        if trials:
            earlyStopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []
        for e in tqdm(range(epoch)):
            train_loss = 0
            for (x, y) in train_loader:
                x.to(self.device)
                y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(x)
            train_loss_list.append(train_loss / len(train_x))

            if trials:
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                valid_loss_list.append(valid_loss.item())
                if not earlyStopper.iscontinuable(valid_metric):
                    break
        if trials:
            self.model.load_state_dict(earlyStopper.best_state)
            plt.plot(valid_loss_list, label='valid loss')

        print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_x, train_y)))

        if trials:
            print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_x, valid_y)))
        print(train_loss_list)
        plt.plot(train_loss_list, label='train_loss')
        plt.legend()
        plt.title(name + ' Training Process')
        plt.savefig('./weight/{:s}.jpg'.format(name))
        plt.show()
        torch.save(self.model.state_dict(), './weight/{:s}.pkl'.format(name))


    def test(self, test_x, test_y):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(test_x)
            test_loss = self.criterion(y_pred, test_y)

            if self.task == 'classification':
                test_metric = metrics.roc_auc_score(test_y.cpu().numpy(), y_pred.cpu().numpy())
            elif self.task == 'regression':
                test_metric = -test_loss

        return test_loss, test_metric








