import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import cosine
import time 
import os
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# 参数设置
data_dir = '/home/sda/luzhixing/datasets/imagenet'  # ImageNet 数据集根目录
batch_size = 224
num_epochs = 5
learning_rate = 1e-05
num_classes = 1000  # ImageNet 数据集包含 1000 个类别
L2_lr = 0.01

address ='/home/sda/luzhixing/volume/CNN/result/imagenet/'
name = 'adamW'+\
        '_batch_size_'+str(batch_size)+\
        '_learning_rate_'+str(learning_rate) +\
          '_num_epochs_'+ str(num_epochs) +\
        '_L2_lr_'+ str(L2_lr)
        
address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
address_3 = address+name+'_diag.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file3 = open(address_3,'w')

def seed_torch(seed=42):
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

def cos_similarity_conv(name,param):
    #print(f"Parameter name: {name}")
    file1.writelines(f" {name}"+'  ')
    #print(f"Parameter value: {param.data.size()}")
    cos_sim_row = cos_sim_column = 0.0
    param = param.cpu()
    for i in range(param.size(2)):
        for j in range(param.size(3)):
            param_i_j = param.data[:,:,i,j]
            cos_sim_row = cos_sim_row + cos_similarity_matrix_row(param_i_j .cpu().data)
            cos_sim_column =cos_sim_column + cos_similarity_matrix_column(param_i_j .cpu().data) 
    
    cos_sim_row = Mean(cos_sim_row) /(param.size(2)*param.size(3))
    cos_sim_column = Mean(cos_sim_column) /(param.size(2)*param.size(3))
    mean_cos_sim_row = round(cos_sim_row,6)
    mean_cos_sim_column= round(cos_sim_column,6)
    
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')

def cos_similarity_ffn(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+'\n')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+','+ str(mean_cos_sim_column)+'\n')
    file1.writelines('='*50+'\n')
    print('='*50)
def volume(matrix):
    #matrix_2d =  param.data.reshape(matrix.size(0),param.data.size(1)*param.data.size(2)*param.data.size(3))
    matrix_2d =  matrix.cpu().detach().numpy()[:,:,1,1]
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


# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用 torchvision.datasets.ImageNet 类加载数据集

train_dataset = datasets.ImageNet(root=data_dir+'/train', split='train', transform=data_transforms['train'])
val_dataset = datasets.ImageNet(root=data_dir+'/val', split='val', transform=data_transforms['val'])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 使用预训练的 ResNet 模型
model = models.resnet50(pretrained=True)

# 修改 ResNet 最后一层，适应 ImageNet 1000 个类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=L2_lr)

# 学习率调度器
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#for name, param in model.named_parameters():
 #   print(f'{name}: {param.size()}')

for name, param in model.named_parameters():
    if name == 'layer4.2.conv2.weight':
        print(f'{name}: {param.size()}')
        file1.writelines('epoch:'+str(0)+'  ')
        cos_similarity_conv(name,param)
        volume(param.cpu().data)
file1.writelines('\n') 
file3.writelines('\n')
     

# 训练和验证函数
def train_model(model, criterion, optimizer, num_epochs=25):
    start_time = time.time() 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_start_time = time.time()
        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 数据迭代
            for i, (inputs, labels) in (enumerate(train_loader) if phase == 'train' else enumerate(val_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if (i+1) % 1000 == 0:
                            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                                    (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item()))
                            file2.writelines('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                                    (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item())+'\n')

                            file1.writelines('epoch:'+str(epoch+1)+'  ')
                            file3.writelines('epoch:'+str(epoch+1)+'  ')
                            for NAME, param in model.named_parameters():
                                if NAME == 'layer4.2.conv2.weight':
                                    print('---------------------------------------------')
                                    print(f'{NAME}: {param.size()}')
                                    cos_similarity_conv(NAME,param)
                                    volume(param.cpu().data)
                                    file1.writelines('\n') 
                                    file3.writelines('\n')

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    

            epoch_loss = running_loss / len(train_dataset if phase == 'train' else val_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset if phase == 'train' else val_dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            file2.writelines(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'+'\n')

        # 计算当前 epoch 所用时间
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time

        # 估算剩余时间
        remaining_time = elapsed_time / (epoch + 1) * (num_epochs - (epoch + 1))
        print(f'Epoch Time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
        print(f'Elapsed Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        print(f'Estimated Remaining Time: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s')
    
    
    total_training_time = time.time() - start_time
    print(f'Training complete in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s')

    return model

# 开始训练
model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# 保存模型
torch.save(model.state_dict(), 'resnet50_imagenet.pth')

file1.close()
file2.close()
file3.close()
print("Training finished.")