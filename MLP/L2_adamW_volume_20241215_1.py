
import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
torch.cuda.set_device(1)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

input_size = 784
batch_size = 512
num_epochs = 200
learning_rate = 0.001
hidden_size = 500
number_H =5
probability = 0.0
random_seed = 42
epoch_interval = 5
Weight_Decay_L2 = 0.01
initial_method = "xavier"  # xavier,kaiming,normal,uniform,orthogonal,constant

address ='/home/sda/luzhixing/volume/MLP/result/'
name = 'L2_'+'xavier_'+'AdamW_'+\
        'B_'+str(batch_size)+\
        '_epoch_'+str(num_epochs)+\
        '_lr_'+ str(learning_rate) +\
        '_width_' + str(hidden_size) +\
         '_seed_'+ str(random_seed )+\
        '_layers_' + str(number_H+1) +\
        '_L2_' + str(Weight_Decay_L2)
address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
address_3 = address+name+'_diag.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file3 = open(address_3,'w')

def seed_torch(seed=random_seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()

def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out
def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix
def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix
def initialize_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif init_type == "uniform":
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_type == "constant":
                nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def cos_similarity(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+'  ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print('PII:'+str(mean_cos_sim_column)+'  ')
    file1.writelines('PII'+','+ str(mean_cos_sim_column)+',')
def volume(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    diagonal_elements = np.diag(Gram_matrix)

    diag_sum = diagonal_elements.sum()
    diag_mean = diagonal_elements.mean()
    diag_var = diagonal_elements.var()

    log_diagonal_elements = np.log(diagonal_elements[diagonal_elements > 0])
    Volume = np.sum(log_diagonal_elements)
    Rank = np.linalg.matrix_rank(Gram_matrix)
    print('Diag_sum:'+str(diag_sum)+'  '+'V:'+str(Volume)+'  '+'Rank:'+str(Rank)+'  ')
    print('='*50)
    file1.writelines('sum_mean_var'+','+str(diag_sum)+','+str(diag_mean)+','+str(diag_var)+\
                     ','+'log_V'+','+str(Volume)+','+'Rank'+','+str(Rank)+',')
    file3.writelines(f"{name}"+','+','.join(f'{x:.5g}' for x in diagonal_elements) + ',')


train_datasets = dsets.MNIST(root ='/home/sda/luzhixing/datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root ='/home/sda/luzhixing/datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)

class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.dropout = nn.Dropout(probability)
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(probability)
        self.norm = nn.BatchNorm1d(hidden)

    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            x = self.r(self.linearH[i](x))

        out = self.out(x)
        return out


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()

initialize_weights(model, init_type=initial_method)    # xavier,kaiming,normal,uniform,orthogonal,constant
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = Weight_Decay_L2)
optimizer.zero_grad()


file1.writelines('epoch:'+str(0)+'  ')
file3.writelines('epoch:'+str(0)+'  ')
for name, param in model.named_parameters():

    if name == 'linearH.0.weight':
        cos_similarity(name)
        volume(param.cpu().data)
    if name == 'linearH.1.weight':
        cos_similarity(name)
        volume(param.cpu().data)
    if name == 'linearH.2.weight':
        cos_similarity(name)
        volume(param.cpu().data)
    if name == 'linearH.3.weight':
        cos_similarity(name)
        volume(param.cpu().data)        
    if name == 'linearH.4.weight':
        cos_similarity(name)
        volume(param.cpu().data) 
file1.writelines('\n') 
file3.writelines('\n')

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        features = Variable(features.view(-1, 28*28)).cuda()
        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs= model(images)
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    print(str(text_accuracy)+'  '+str(loss_out))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'\n')
    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        file3.writelines('epoch:'+str(epoch)+'  ')
        
        for name, param in model.named_parameters():   
            if name == 'linearH.0.weight':
                cos_similarity(name)
                volume(param.cpu().data)
            if name == 'linearH.1.weight':
                cos_similarity(name)
                volume(param.cpu().data)
            if name == 'linearH.2.weight':
                cos_similarity(name)
                volume(param.cpu().data)
            if name == 'linearH.3.weight':
                cos_similarity(name)
                volume(param.cpu().data)        
            if name == 'linearH.4.weight':
                cos_similarity(name)
                volume(param.cpu().data)
        file1.writelines('\n')
        file3.writelines('\n')


file1.close() 
file2.close()
file3.close()






