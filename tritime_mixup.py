import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
import random
import numpy as np
import os
import torch
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(2023)

dataset = "Yoga"

np.random.seed(2023)

x, y = load_classification(dataset)

porc_unlabel = 0.6


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2023, stratify=y)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

x_label, x_unlabel, y_label, y_unlabel, label_indices, unlabel_indices = train_test_split(x_train, y_train, np.arange(len(y_train)), test_size=porc_unlabel, random_state=10)

x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[2], 1])
x_label = np.reshape(x_label, [x_label.shape[0], x_label.shape[2], 1])
x_unlabel = np.reshape(x_unlabel, [x_unlabel.shape[0], x_unlabel.shape[2], 1])
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[2], 1])


# Inicializar y_fake con ceros del mismo tamaño que y_train
y_fake = np.zeros_like(y_train)

# Asignar 1 a y_fake en las posiciones de los índices etiquetados
y_fake[label_indices] = 1



def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

import torch.nn as nn

class TimeSeriesConvolution(nn.Module):
    def __init__(self, in_channels, num_class):
        super(TimeSeriesConvolution, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, 8, padding="same")
        self.batch1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 5, padding="same")
        self.batch2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 3, padding="same")
        self.batch3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, 3, padding="same")
        self.batch4 = nn.BatchNorm1d(64)
        self.batch5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc_multiclass = nn.Linear(64, num_class)
        self.relu = nn.ReLU()
        self.pretext = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)




    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv1 = self.batch1(self.relu(self.conv1(x)))
        #conv1 = self.relu(self.batch1(self.conv1(x)))
        conv2 = self.batch2(self.relu(self.conv2(conv1)))
        #conv2 = self.relu(self.batch2(self.conv2(conv1)))
        conv3 = self.batch3(self.relu(self.conv3(conv2)))
        #conv3 = self.relu(self.batch3(self.conv3(conv2)))
        conv4 = self.batch4(self.relu(self.conv4(conv3)))
        #conv4 = self.relu(self.batch4(self.conv4(conv3)))
        global_pooled = self.global_pooling(conv4)
        flattened = global_pooled.view(global_pooled.size(0), -1)
        output_multiclass = self.fc_multiclass(self.batch5(flattened))


        return flattened, self.relu(output_multiclass)


import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F



class TimeSeriesDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx].astype(np.float32)
        label = self.label[idx]
        return torch.tensor(series, dtype=torch.float32), label

def noise_transform(series, i=0, noise_std=0.1):
    np.random.seed(i)
    noisy_series = series + np.random.normal(0, noise_std, size=series.shape[1])
    return noisy_series

class TimeSeriesDataset2(Dataset):
    def __init__(self, data, label, fake_label, apply_transform):
        self.data = data
        self.label = label
        self.fake_label = fake_label
        self.apply_transform = apply_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx].astype(np.float32)
        label = self.label[idx]
        fake_label = self.fake_label[idx]
        if self.apply_transform:
            flip_series = series[::-1].copy()

            #flip_series = series.copy()
            noise_series2 = noise_transform(series, 1)
            #noise_series2 = series.copy()
            #return torch.tensor(series), torch.tensor(noise_series2, dtype=torch.float32), torch.tensor(flip_series, dtype=torch.float32), label, fake_label
            return torch.tensor(series),  label, fake_label
        else:
            return torch.tensor(series, dtype=torch.float32), label


from tqdm import tqdm

dataset_train = TimeSeriesDataset2(data=x_train, label=y_train, fake_label=y_fake, apply_transform=True)
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)


dataset_label = TimeSeriesDataset(data=x_label, label=y_label)
dataloader_label = DataLoader(dataset_label, batch_size=64, shuffle=True)



dataset_test = TimeSeriesDataset(data=x_test, label=y_test)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)



class TripletLossEuclidean(nn.Module):
    def __init__(self, margin=20.0):
        super(TripletLossEuclidean, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculamos la distancia euclidiana entre las incrustaciones
        pos_euclidean_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_euclidean_dist = F.pairwise_distance(anchor, negative, p=2)

        # Calculamos la pérdida Triplet con distancia euclidiana
        loss = torch.relu(self.margin + pos_euclidean_dist - neg_euclidean_dist).mean()

        return loss




def compute_cv_without_labels(features):
    # Convertir a numpy array si es necesario
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    # Asegúrate de que features es de la forma [n_samples, n_timestamps, 1]
    features = features.squeeze(axis=2)  # [n_samples, n_timestamps]

    # Inicializar lista de distancias
    distances = []
    
    # Calcular las distancias euclidianas entre todos los pares de series
    n_samples = features.shape[0]
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = np.sqrt(np.sum((features[i] - features[j]) ** 2))
            distances.append(distance)

    # Convertir distancias a un array numpy
    distances = np.array(distances)

    # Calcular media y desviación estándar de las distancias
    mean_distance = np.mean(distances) if len(distances) > 0 else 0
    std_distance = np.std(distances) if len(distances) > 0 else 0

    # Calcular el coeficiente de variación (CV)
    cv = std_distance / mean_distance if mean_distance != 0 else 0
    
    return cv


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

all_accuracy, all_loss = [], []
all_recall, all_precision, all_f1 = [], [], []

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loop para tunear alpha
best_alpha = 0.1
best_f1 = 0.0

for alpha in torch.arange(0.1, 1.0, 0.1):
    print("MODELO CON ALPHA:", alpha)
    all_recall, all_precision, all_f1 = [], [], []
    all_accuracy, all_loss = [], []
    for seed in range(10):


      print("model", seed)
      print("Starting Training")
      print("--------------------------------------------------------------------------")
      set_seed(seed)

      backbone = TimeSeriesConvolution(in_channels=x_train.shape[2], num_class=len(np.unique(y_train)))



      backbone.to(device)


      criterion1 = nn.CrossEntropyLoss()
      criterion2 = TripletLossEuclidean()


      optimizer = torch.optim.Adam(backbone.parameters())
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
      factor=0.5, patience=50, threshold=0.0001, min_lr=0.0001)


      epochs1, epochs2 = 800, 1200

      backbone.train()
 

      for epoch in range(epochs1):
          total_loss = 0
          total_loss_label = 0
          total_loss_unlabel = 0
          for batch in tqdm(dataloader_train):
            optimizer.zero_grad()
            x1 = batch[0]
            y = batch[1]
            y_fake = batch[2]    

   
            x = x1.numpy()
        

            x3 = np.flip(x).copy()
       
       
        
            x3 = alpha * x1 + (1-alpha) * x3
        
            x3 = x3.clone().detach().float()
       

            x2 = x + np.random.normal(0, 0.1, size=x.shape)
       
            x2 = torch.tensor(x2, dtype=torch.float32).clone().detach()


            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y_fake = y_fake.to(device)

            y = y.to(device, dtype=torch.long)
       

            anchor, y_pred = backbone(x1)
            positive, y_pred2 = backbone(x2)
            negative, y_pred3 = backbone(x3)


            loss =  criterion2(anchor, positive, negative)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()


          avg_loss = total_loss / len(dataloader_train)

          scheduler.step(avg_loss)
          print(f"epoch: {epoch:>02},loss: {avg_loss}, ")

      for epoch in range(epochs2):
          total_loss = 0
          for batch in tqdm(dataloader_train):
            optimizer.zero_grad()
            x1 = batch[0]


            y = batch[1]
            y_fake = batch[2]

            x1 = x1.to(device)
            y_fake = y_fake.to(device)

            y = y.to(device, dtype=torch.long)

            anchor, y_pred = backbone(x1)

            # Seleccionar solo los elementos donde y_fake es 1
            selected_y_pred = y_pred[y_fake == 1]
            selected_y_true = y[y_fake == 1]

            loss = criterion1(selected_y_pred, selected_y_true)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()


          avg_loss = total_loss / len(dataloader_train)

          scheduler.step(avg_loss)
          print(f"epoch: {epoch:>02},loss: {avg_loss} ")



      backbone.eval()
      total_test_loss = 0
      total_test_acc = 0
      y_true_test = []
      y_pred_test = []
      all_vectors = []
      all_labels = []

      with torch.no_grad():
        for inputs, labels in dataloader_test:
          labels = labels.type(torch.LongTensor)
          inputs, labels = inputs.to(device), labels.to(device)
          vector, outputs = backbone(inputs)

          predicted = torch.softmax(outputs, dim=1).argmax(dim=1)

          test_acc = accuracy_fn(y_true=labels, y_pred=predicted)
          total_test_acc += torch.tensor(test_acc).item()

          y_true_test.extend(labels.cpu().numpy())
          y_pred_test.extend(predicted.cpu().numpy())
          all_vectors.append(vector.cpu())
          all_labels.append(labels.cpu())

      test_accuracy = accuracy_score(y_true_test, y_pred_test)
      test_recall = recall_score(y_true_test, y_pred_test, average="macro")
      test_precision = precision_score(y_true_test, y_pred_test, average="macro")
      test_f1 = f1_score(y_true_test, y_pred_test, average="macro")


      all_accuracy.append(test_accuracy)
      all_recall.append(test_recall)
      all_precision.append(test_precision)
      all_f1.append(test_f1)
      print("test_accuracy", test_accuracy)






    print("all_accuracy promedio: " ,np.mean(np.array(all_accuracy)))
    print("all_accuracy std: " ,np.std(np.array(all_accuracy)))



    print("all_precision promedio: " ,np.mean(np.array(all_precision)))
    print("all_precision std: " ,np.std(np.array(all_precision)))


    print("all_f1 promedio: " ,np.mean(np.array(all_f1)))
    print("all_f1 std: " ,np.std(np.array(all_f1)))


    print("all_recall promedio: " ,np.mean(np.array(all_recall)))
    print("all_recall sd: " ,np.std(np.array(all_recall)))


    if np.mean(np.array(all_f1)) > best_f1:
      best_f1 = np.mean(np.array(all_f1))
      std_f1 = np.std(np.array(all_f1))

      best_accuracy = np.mean(np.array(all_accuracy))
      std_accuracy = np.std(np.array(all_accuracy))

      best_recall = np.mean(np.array(all_recall))
      std_recall = np.std(np.array(all_recall))

      best_precision = np.mean(np.array(all_precision))
      std_precision = np.std(np.array(all_precision))
      best_alpha = alpha.item()




print(f'Mejor alpha: {best_alpha:.3f}')

print("accuracy promedio: " ,best_accuracy)
print("accuracy std: " ,std_accuracy)


print("precision promedio: " ,best_precision)
print("precision std: " ,std_precision)

print("f1 promedio: " ,best_f1)
print("f1 std: " ,std_f1)

print("recall promedio: " ,best_recall)
print("recall std: " ,std_recall)

print(dataset, porc_unlabel)
