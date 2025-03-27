import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine

torch.cuda.set_device(1)


address ='/home/sda/luzhixing/volume/PLM/result/llama/'

address_1 = address+'llama1_7b'+'_volume.txt'


file1 = open(address_1,'w')


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



def volume(matrix):
    gram_matrix = torch.matmul(matrix.T, matrix)
    diagonal = torch.diag(gram_matrix)
    diag_sum = torch.sum(diagonal).item()
    diag_mean = torch.mean(diagonal).item()
    diag_var = torch.var(diagonal, unbiased=False).item()

    print('Diag_sum:'+str(diag_sum)+'  '+'diag_mean:'+str(diag_mean)+' '+'diag_var:'+str(diag_var))
    print('='*50)
    file1.writelines('sum_mean_var'+','+str(diag_sum)+','+str(diag_mean)+','+str(diag_var)+'\n')




# 模型路径
model_path = "/home/sda/luzhixing/datasets/llama/llama1-7b"

# 加载分词器
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# 加载模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 根据需要调整数据类型
    device_map="auto"           # 自动选择设备（GPU/CPU）
)





for name, param in model.named_parameters():
    for i in range(0,32,1):
        if name == f'model.layers.{i}.self_attn.q_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.self_attn.k_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.self_attn.v_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.self_attn.o_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.mlp.gate_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.mlp.up_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)
        if name == f'model.layers.{i}.mlp.down_proj.weight':
            parts = name.split('.')
            file1.writelines( f"{parts[2]}_{parts[4]}"+',')
            volume(param.data)




file1.close() 



#for name, param in model.named_parameters():
  #  print(f"参数名称: {name}, 参数大小: {param.size()}")