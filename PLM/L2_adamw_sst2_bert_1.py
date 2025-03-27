from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数
num_epochs = 20
learning_rate = 2e-5
random_seed = 42
Batch_size = 256
Weight_Decay_L2 = 0.1
#tasks = ['mrpc', 'qqp', 'qnli', 'stsb', 'rte','sst2', 'cola' ]

"""
task_to_keys = {
"cola": ("sentence", None),
"mnli": ("premise", "hypothesis"),
"mrpc": ("sentence1", "sentence2"),
"qnli": ("question", "sentence"),
"qqp": ("question1", "question2"),
"rte": ("sentence1", "sentence2"),
"sst2": ("sentence", None),
"stsb": ("sentence1", "sentence2"),
"wnli": ("sentence1", "sentence2"),
}

"""

Task = 'sst2'

address ='/home/sda/luzhixing/volume/PLM/result/'
name = 'L2_'+'adamW_'+str(Task)+\
        '_lr_'+str(learning_rate) +\
         '_seed_'+ str(random_seed )+\
          '_epochs_'+ str(num_epochs) +\
           '_L2_'+ str(Weight_Decay_L2)
address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
address_3 = address+name+'_diag.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file3 = open(address_3,'w')

###############################
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
def column_block_matrix(matrix, block_size):
    n, m = matrix.shape
    num_blocks = n // block_size
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        block = matrix[start:end,:]
        blocks.append(block)
    return blocks
def cos_similarity(name):
    print(f"Parameter name: {name}")
    #file1.writelines(f"{name}"+'  ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    print(mean_cos_sim_row)
    #file1.writelines(str(mean_cos_sim_row)+'  ')
    return mean_cos_sim_row


def seed_torch(seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
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
    file3.writelines(f"{parts[3]}_{parts[6]}"+','+','.join(f'{x:.5g}' for x in diagonal_elements) + ',')
##############################################


seed_torch(random_seed)
# 定义保存数据集的函数
def save_dataset(dataset, task):
    dataset.save_to_disk(f'/home/sda/luzhixing/datasets/glue_dataset/{task}')

# 定义加载数据集的函数
def load_or_download_dataset(task):
 
    dataset_path = f'/home/sda/luzhixing/datasets/glue_dataset/{task}'
    if os.path.exists(dataset_path):
        return load_from_disk(dataset_path)
    else:
        dataset = load_dataset('glue', task)
        save_dataset(dataset, task)
        return dataset
# 加载数据集
dataset = load_or_download_dataset(Task)

train_dataset = dataset['train']
val_dataset = dataset['validation']  # 加载验证集

# 加载模型和tokenizer
# 定义加载或保存模型的函数
def load_or_save_model(model_name):
    model_path = f'/home/sda/luzhixing/pretrained_model/bert_models/{model_name}'
    if os.path.exists(model_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
        model.save_pretrained(model_path)
        return model

model = load_or_save_model('bert-base-cased')  # 加载或保存模型


#model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 编码函数
def encode(examples):
    #return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
    return tokenizer(examples['sentence'],  truncation=True, padding='max_length', max_length=128)
    #return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

# 对训练集和验证集进行编码
train_dataset = train_dataset.map(encode, batched=True)
val_dataset = val_dataset.map(encode, batched=True)

# 添加标签
train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

# 格式化为torch tensor
train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

# 创建dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=Batch_size)

# 将模型移动到设备
model.train().to(device)
#for name, param in model.named_parameters():
  #  print(f'{name}: {param.size()}')

optimizer = torch.optim.AdamW(params=model.parameters(), lr = learning_rate, weight_decay = Weight_Decay_L2)


for name, param in model.named_parameters():
        for i in range(0,12,2):
            if name == f'bert.encoder.layer.{i}.attention.self.query.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data)
            if name == f'bert.encoder.layer.{i}.attention.self.key.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data)
            if name == f'bert.encoder.layer.{i}.attention.self.value.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data) 
file1.writelines('\n') 
file3.writelines('\n') 


# 训练
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss  # 计算损失
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")
            file2.writelines(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}"+'\n')
            
    for name, param in model.named_parameters():
        for i in range(0,12,2):
            if name == f'bert.encoder.layer.{i}.attention.self.query.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data)
            if name == f'bert.encoder.layer.{i}.attention.self.key.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data)
            if name == f'bert.encoder.layer.{i}.attention.self.value.weight':
                parts = name.split('.')
                file1.writelines( f"{parts[3]}_{parts[6]}"+','+str(cos_similarity(name))+',')
                volume(param.cpu().data)      
    file1.writelines('\n') 
    file3.writelines('\n') 


    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average training loss: {avg_train_loss:.4f}")

    # 评估模型在验证集上的表现
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    val_accuracy = correct / total
    #print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.4f}"+'  '+'Lr:'+str(round(scheduler.get_last_lr()[0],6)))
    print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.4f}")
    file2.writelines(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.4f}"+'\n')

# 保存模型
model.save_pretrained('./bert-qqp-model')
tokenizer.save_pretrained('./bert-qqp-tokenizer')
print("Model and tokenizer saved.")

file1.close() 
file2.close()
file3.close()

