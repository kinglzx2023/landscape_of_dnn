import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
torch.cuda.set_device(1)

Batch_size = 512
num_epochs = 40
learning_rate = 0.001
random_seed = 42
epoch_interval = 4
probability = 0.0
Weight_Decay_L2 = 0.0
initial_method = "xavier"  # xavier,kaiming,normal,uniform,orthogonal,constant

address ='/home/sda/luzhixing/volume/CNN/result/cifar10/'

name =  'ceshi_'+'xavier_'+'AdamW_'+'B_'+str(Batch_size)+\
        '_epoch_'+str(num_epochs)+\
        '_lr_'+ str(learning_rate)  +\
        '_seed_'+ str(random_seed ) +\
        '_pro_' + str(probability)  +\
        '_L2_' + str(Weight_Decay_L2) 

address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
address_3 = address+name+'_diag.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file3 = open(address_3,'w')


def seed_torch(seed=random_seed):
    #random.seed(seed)
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

def cos_similarity_conv(name):
    print(f"Parameter name: {name}")
    file1.writelines(f" {name}"+'  ')
    print(f"Parameter value: {param.data.size()}")
    cos_sim_row = cos_sim_column = 0.0
    for i in range(param.size(2)):
        for j in range(param.size(3)):
            param_i_j = param.data[:,:,i,j]
            cos_sim_row = cos_sim_row + cos_similarity_matrix_row(param_i_j .cpu().data)
            cos_sim_column =cos_sim_column + \
                cos_similarity_matrix_column(param_i_j .cpu().data) 
    
    cos_sim_row = Mean(cos_sim_row) /(param.size(2)*param.size(3))
    cos_sim_column = Mean(cos_sim_column) /(param.size(2)*param.size(3))
    mean_cos_sim_row = round(cos_sim_row,6)
    mean_cos_sim_column= round(cos_sim_column,6)
    
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines('PII'+' '+str(mean_cos_sim_row)+' '+ str(mean_cos_sim_column)+' ')
    print('='*50)

def volume(matrix):
    
    #matrix_2d = matrix[:,:,1,1]
    matrix_2d =  param.data.reshape(matrix.size(0),param.data.size(1)*param.data.size(2)*param.data.size(3))
    matrix_2d = matrix_2d.cpu()
    matrix_transpose = np.transpose(matrix_2d)
    Gram_matrix = np.dot(matrix_transpose,matrix_2d)
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





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/sda/luzhixing/datasets/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/sda/luzhixing/datasets/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()

initialize_weights(model, init_type=initial_method)  
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=Weight_Decay_L2)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = Weight_Decay_L2)

file1.writelines('epoch:'+str(0)+'  ')
file3.writelines('epoch:'+str(0)+'  ')
for name, param in model.named_parameters():
    if name == 'layer1.0.conv1.weight':
        cos_similarity_conv(name)
        volume(param.cpu().data)   
    if name == 'layer2.0.conv1.weight':
        cos_similarity_conv(name)
        volume(param.cpu().data)
    if name == 'layer3.0.conv1.weight':
        cos_similarity_conv(name)
        volume(param.cpu().data)
    if name == 'layer4.0.conv1.weight':
        cos_similarity_conv(name)
        volume(param.cpu().data)
        file1.writelines('\n')
        file3.writelines('\n')



for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = total_loss / len(trainloader)
    train_accuracy = correct / total

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    file2.writelines(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}'+'\n')
    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        file3.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters(): 
            if name == 'layer1.0.conv1.weight':
                cos_similarity_conv(name)
                volume(param.cpu().data)
            if name == 'layer2.0.conv1.weight':
                cos_similarity_conv(name)
                volume(param.cpu().data)
            if name == 'layer3.0.conv1.weight':
                cos_similarity_conv(name)
                volume(param.cpu().data)
            if name == 'layer4.0.conv1.weight':
                cos_similarity_conv(name)
                volume(param.cpu().data)
                file1.writelines('\n')
                file3.writelines('\n')

file1.close()
file2.close()
file3.close()
print("Training finished.")
