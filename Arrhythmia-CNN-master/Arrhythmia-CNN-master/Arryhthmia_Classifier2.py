from __future__ import print_function
import torch
import torch.utils.data
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score

file_path = './data/Arrhythmia_dataset.pkl'
df = pd.read_pickle(file_path)

#label_column = 'target'
#issing_label_mask = df[label_column].isna()
df = df.dropna()
print("Data Cleaned")

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)

# Save the updated datasets
train_file_path = './data/Arrhythmia_train_dataset.pkl'
test_file_path = './data/Arrhythmia_test_dataset.pkl'
train_df.to_pickle(train_file_path)
test_df.to_pickle(test_file_path)

print("Data split into training and testing sets")
##########################################################################################
is_cuda = False
num_epochs = 5
batch_size = 10
torch.manual_seed(46)
log_interval = 10
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600
num_records = 48
num_classes = 16
allow_label_leakage = True

device = torch.device("cuda:2" if is_cuda else "cpu")
# train_ids, test_ids = train_test_split(np.arange(index_set), train_size=.8, random_state=46)
# scaler = MinMaxScaler(feature_range=(0, 1), copy=False)

class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_path, transforms_=None):
        self.df = pd.read_pickle(data_path)
        self.transforms = transforms_

    def __getitem__(self, index):
        row = self.df.iloc[index]
        signal = row['signal']
        target = row['target']
        if self.transforms is not None:
            signal = self.transforms(signal)
        signal = signal.reshape(1, signal.shape[0])
        return signal, target

    def __len__(self):
        return self.df.shape[0]

train_dataset = CustomDatasetFromCSV('./data/Arrhythmia_train_dataset.pkl')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CustomDatasetFromCSV('./data/Arrhythmia_test_dataset.pkl')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def basic_layer(in_channels, out_channels, kernel_size, batch_norm=False, max_pool=True, conv_stride=1, padding=0
                , pool_stride=2, pool_size=2):
    layer = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride,
                  padding=padding),
        nn.ReLU())
    if batch_norm:
        layer = nn.Sequential(
            layer,
            nn.BatchNorm1d(num_features=out_channels))
    if max_pool:
        layer = nn.Sequential(
            layer,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))

    return layer

class arrhythmia_classifier(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(arrhythmia_classifier, self).__init__()
        self.cnn = nn.Sequential(
            basic_layer(in_channels=in_channels, out_channels=128, kernel_size=50, batch_norm=True, max_pool=True,
                        conv_stride=3, pool_stride=3),
            basic_layer(in_channels=128, out_channels=32, kernel_size=7, batch_norm=True, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=32, out_channels=32, kernel_size=10, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=32, out_channels=128, kernel_size=5, batch_norm=False, max_pool=True,
                        conv_stride=2, pool_stride=2),
            basic_layer(in_channels=128, out_channels=256, kernel_size=15, batch_norm=False, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=256, out_channels=512, kernel_size=5, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=512, out_channels=128, kernel_size=3, batch_norm=False, max_pool=False,
                        conv_stride=1),
            Flatten(),
            nn.Linear(in_features=1152, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(dim=1)  # Specify the dimension for softmax
        )

    def forward(self, x, ex_features=None):
        return self.cnn(x)


def calc_next_len_conv1d(current_len=112500, kernel_size=16, stride=8, padding=0, dilation=1):
    return int(np.floor((current_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


model = arrhythmia_classifier().to(device).double()
lr = 0.0003
num_of_iteration = len(train_dataset) // batch_size

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.NLLLoss()


def train(epoch):
    model.train()
    train_loss = 0
    predictions = []
    targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        predictions.extend(output.argmax(dim=1).cpu().numpy())
        targets.extend(target.cpu().numpy())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    """ precision = precision_score(targets, predictions, average='macro')
    accuracy = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, average='macro') """
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    """ print(f'Precision: {precision:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}') """


def test(epoch):
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(target.cpu().numpy())
            if batch_idx == 0:
                n = min(data.size(0), 4)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.5f}'.format(test_loss))
    
    """ precision = precision_score(targets, predictions, average='macro', zero_division=1)
    recall = recall_score(targets, predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(targets, predictions)
    
    print(f'Precision: {precision:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}')
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}') """
    
    # Filter out classes with no predicted samples
    classes_with_no_predictions = set(targets) - set(predictions)
    targets_filtered = [t for t, p in zip(targets, predictions) if p not in classes_with_no_predictions]
    predictions_filtered = [p for p in predictions if p not in classes_with_no_predictions]

    # Calculate precision, accuracy, and recall using the filtered predictions and targets
    precision_filtered = precision_score(targets_filtered, predictions_filtered, average='macro', zero_division=1)
    recall_filtered = recall_score(targets_filtered, predictions_filtered, average='macro', zero_division=1)
    accuracy_filtered = accuracy_score(targets_filtered, predictions_filtered)
    
    print(f'Filtered Precision: {precision_filtered:.4f}, Filtered Accuracy: {accuracy_filtered:.4f}, Filtered Recall: {recall_filtered:.4f}')
    
if __name__ == "__main__":
    for epoch in range(1, num_epochs):
        train(epoch)
        test(epoch)