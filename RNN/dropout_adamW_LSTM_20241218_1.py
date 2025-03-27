# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import numpy as np
import torch.onnx
torch.cuda.set_device(1)
from scipy.spatial.distance import cosine
import torchvision.transforms as transforms
from torch.autograd import Variable

import data
import model
random_seed = 42
Weight_decay = 0.02


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model_PII_LSTM.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=4,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()


address ='/home/sda/luzhixing/volume/RNN/result/'
name = 'dropout_LSTM_'+ 'adamW' +\
        '_lr_'+ str(args.lr)+\
        '_epoch_'+ str(args.epochs)+\
        '_batch_'+str(args.batch_size)+\
        '_emsize_' +str(args.emsize) +\
        '_nhid_' +str(args.nhid) +\
        '_nlayers_' + str(args.nlayers) +\
        '_nhid_'+ str(args.nhid)+\
        '_nlayers_'+str(args.nlayers) +\
        '_drop_'+str(args.dropout)
         

address_1 = address+name+'_volume.txt'
address_2 = address+name+'_acc.txt'
address_3 = address+name+'_diag.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')
file3 = open(address_3,'w')



################################
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

Matrix = ['Input_gate','Forget_gate','Cell','Output_gate']

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


def cos_similarity_LSTM(index,block,parts):
    cos_sim_row = cos_similarity_matrix_row(block)
    cos_sim_column = cos_similarity_matrix_column(block) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(parts+','+str(Matrix[index])+','+str(mean_cos_sim_row)+','+ str(mean_cos_sim_column)+',')

def volume(index,matrix,parts):
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
    file3.writelines(parts+','+str(Matrix[index])+','+','.join(f'{x:.5g}' for x in diagonal_elements) + ',')
 
##############################################


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

corpus = data.Corpus(args.data)


# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()




optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr,weight_decay = Weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
#for name, param in model.named_parameters():
 #   print(f'{name}: {param.size()}')
###############################################################################
# Training code
###############################################################################
file1.writelines('epoch:'+str(0)+',')
file3.writelines('epoch:'+str(0)+',')
for name, param in model.named_parameters():
    if name == 'rnn.weight_hh_l0':
        parts = name.split('weight_')[-1]
        LSTM_gates = column_block_matrix(param.cpu().data, 500)
        for index,block in enumerate(LSTM_gates):
            cos_similarity_LSTM(index,block,parts)
            volume(index,block,parts)
    if name == 'rnn.weight_hh_l1':
        parts = name.split('weight_')[-1]
        LSTM_gates = column_block_matrix(param.cpu().data, 500)
        for index,block in enumerate(LSTM_gates):
            cos_similarity_LSTM(index,block,parts)
            volume(index,block,parts)
    if name == 'rnn.weight_hh_l2':
        parts = name.split('weight_')[-1]
        LSTM_gates = column_block_matrix(param.cpu().data, 500)
        for index,block in enumerate(LSTM_gates):
            cos_similarity_LSTM(index,block,parts)
            volume(index,block,parts)
    if name == 'rnn.weight_hh_l3':
        parts = name.split('weight_')[-1]
        LSTM_gates = column_block_matrix(param.cpu().data, 500)
        for index,block in enumerate(LSTM_gates):
            cos_similarity_LSTM(index,block,parts)
            volume(index,block,parts)
file1.writelines('\n')
file3.writelines('\n')




def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    #for name, param in model.named_parameters():
        #print(f'{name}: {param.size()}')
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break
    file1.writelines('epoch:'+str(epoch)+',')
    file3.writelines('epoch:'+str(epoch)+',')
    for name, param in model.named_parameters():
        if name == 'rnn.weight_hh_l0':
            parts = name.split('weight_')[-1]
            LSTM_gates = column_block_matrix(param.cpu().data, 500)
            for index,block in enumerate(LSTM_gates):
                cos_similarity_LSTM(index,block,parts)
                volume(index,block,parts)
        if name == 'rnn.weight_hh_l1':
            parts = name.split('weight_')[-1]
            LSTM_gates = column_block_matrix(param.cpu().data, 500)
            for index,block in enumerate(LSTM_gates):
                cos_similarity_LSTM(index,block,parts)
                volume(index,block,parts)
        if name == 'rnn.weight_hh_l2':
            parts = name.split('weight_')[-1]
            LSTM_gates = column_block_matrix(param.cpu().data, 500)
            for index,block in enumerate(LSTM_gates):
                cos_similarity_LSTM(index,block,parts)
                volume(index,block,parts)
        if name == 'rnn.weight_hh_l3':
            parts = name.split('weight_')[-1]
            LSTM_gates = column_block_matrix(param.cpu().data, 500)
            for index,block in enumerate(LSTM_gates):
                cos_similarity_LSTM(index,block,parts)
                volume(index,block,parts)
    file1.writelines('\n')
    file3.writelines('\n')

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)

        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        file2.writelines('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)) + '\n')
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        #else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


file1.close()
file2.close()
file3.close()
