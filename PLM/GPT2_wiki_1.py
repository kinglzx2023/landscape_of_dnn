import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn.functional as F
import os
import pickle
import numpy as np
import math
from scipy.spatial.distance import cosine
from tqdm import tqdm

# 设置设备

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# input_size = 32 * 32 * 3
batch_size = 18
num_epochs = 10
learning_rate = 2e-5
random_seed = 42

epoch_interval = 1
save_interval = 50
l2 = 0.1          #1e-4


address ='/home/sda/luzhixing/volume/PLM/result/gpt-2/'
name = 'GPT2'+'_lr_'+ 'adamW_'+\
        '_lr_'+ str(learning_rate) +\
        '_batch_'+str(batch_size)+\
        '_epochs_'+ str(num_epochs) +\
        '_L2_' + str(l2)

address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')

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


Index = ['Q','K','V']
def volume_GPT_atten(m,matrix):
    Q_matrix = matrix[:,0:768]
    K_matrix = matrix[:,768:1536]
    V_matrix = matrix[:,1536:2304]
    
    matrices = [Q_matrix,K_matrix,V_matrix]
    for i, MATRIX in enumerate(matrices):
        matrix_transpose = np.transpose(MATRIX)
        Gram_matrix = np.dot(matrix_transpose,MATRIX)
        diagonal_elements = np.diag(Gram_matrix)
        diag_sum = diagonal_elements.sum()
        diag_mean = diagonal_elements.mean()
        diag_var = diagonal_elements.var()
        print('Diag_sum:'+str(diag_sum)+'  '+'diag_mean:'+str(diag_mean)+'  '+'diag_var:'+str(diag_var)+'  ')
        print('='*50)
        file1.writelines(str(m)+'_'+Index[i]+','+'sum_mean_var'+','+str(diag_sum)+','+str(diag_mean)+','+str(diag_var)+','+'\n')
def cos_similarity(matrix): 
    cos_sim_row = cos_similarity_matrix_row(matrix) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    #file1.writelines(str(mean_cos_sim_row)+'  ')
    return mean_cos_sim_row
def volume(matrix):
    gram_matrix = torch.matmul(matrix.T, matrix)
    diagonal = torch.diag(gram_matrix)
    diag_sum = torch.sum(diagonal).item()
    diag_mean = torch.mean(diagonal).item()
    diag_var = torch.var(diagonal, unbiased=False).item()

    print('Diag_sum:'+str(diag_sum)+'  '+'diag_mean:'+str(diag_mean)+' '+'diag_var:'+str(diag_var))
    print('='*50)
    file1.writelines('sum_mean_var'+','+str(diag_sum)+','+str(diag_mean)+','+str(diag_var)+'\n')
# 加载wikitext-2数据集
dataset = load_dataset("/home/sda/luzhixing/datasets/wikitext-2-raw-v1")
print(dataset)

# dataset = load_dataset('text', data_files={'train': '/home/sda/zhouqin/.data/wikitext-2/wikitext-2/wiki.train.tokens', 'validation': '/home/sda/zhouqin/.data/wikitext-2/wikitext-2/wiki.valid.tokens'})


# 初始化BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('/home/sda/zhouqin/sparity/model/bert-base-uncased/')
tokenizer = GPT2Tokenizer.from_pretrained("/home/sda/luzhixing/datasets/gpt2")
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 进行文本数据预处理
def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# 对数据集应用tokenization
train_dataset = dataset['train'].map(tokenize_function, batched=True)
valid_dataset = dataset['validation'].map(tokenize_function, batched=True)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size)


#model = GPT2LMHeadModel.from_pretrained('/home/sda/zhouqin/sparity/model/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
model = GPT2LMHeadModel.from_pretrained('/home/sda/luzhixing/datasets/gpt2')
model.to(device)

#for name, param in model.named_parameters():
 #   print(name,param.data.size())

for name, param in model.named_parameters():
        for j in range(0,12,1):
            if name == f'transformer.h.{j}.attn.c_attn.weight': 
                volume_GPT_atten(j,param.cpu().data)
            if name == f'transformer.h.{j}.attn.c_proj.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)
            if name == f'transformer.h.{j}.mlp.c_fc.weight': 
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)
            if name == f'transformer.h.{j}.mlp.c_proj.weight': 
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)



def compute_perplexity(loss):
    return torch.exp(loss)


optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate,weight_decay= l2)



for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        # print(batch['labels'])
        input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
        attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
        labels = torch.stack(batch['labels'], dim=1).to(device)

        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_train_loss = total_loss/len(train_dataloader)
    # 验证
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # labels = batch['labels'].to(device)
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
            labels = torch.stack(batch['labels'], dim=1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
    avg_eval_loss = eval_loss / len(eval_dataloader)
    perplexity = compute_perplexity(torch.tensor(avg_eval_loss))
    
    print(f"Epoch {epoch+1}/{num_epochs}, Perplexity: {perplexity:.4f}, Loss: {avg_train_loss:.4f},Validation Loss: {avg_eval_loss:.4f}")
    file2.writelines(f"Epoch {epoch+1}/{num_epochs}, Perplexity: {perplexity:.4f}, Loss: {avg_train_loss:.4f},Validation Loss: {avg_eval_loss:.4f}"+'\n')

file1.writelines('Trained'+'\n')

for name, param in model.named_parameters():
        for j in range(0,12,1):
            if name == f'transformer.h.{j}.attn.c_attn.weight': 
                volume_GPT_atten(j,param.cpu().data)
            if name == f'transformer.h.{j}.attn.c_proj.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)
            if name == f'transformer.h.{j}.mlp.c_fc.weight': 
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)
            if name == f'transformer.h.{j}.mlp.c_proj.weight': 
                parts = name.split('.')
                file1.writelines( f"{parts[2]}_{parts[3]}_{parts[4]}"+',')
                volume(param.data)


# 只保存模型的权重
# torch.save(model.state_dict(), '/home/sda/zhouqin/sparity/model/gpt2_model_weights_1.pth')
 


file1.close() 
file2.close()

