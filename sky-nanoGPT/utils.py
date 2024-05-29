import os
import ast
import pdb
import pickle
from tqdm import tqdm 

import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt

with open('data/enwik8/int2char.pkl', 'rb') as f:
    int2char = pickle.load(f)
    char2int = {v:k for k,v in int2char.items()}
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    consonants = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
    vowelidxs = [char2int[v] for v in vowels]
    consonantidxs = [char2int[c] for c in consonants]

def pos_embed_vowel_consonant(context, embedding_weight:float):
    """We want to create an additional encoding for vowels and consonants, which 
        .. is added to our positional embedding.

        We assign the following
        Vowels      ==>  epsilon
        Consonants  ==> -epsilon
        Spaces = 0 
        
        Args:
            - Context (list-like) (BS x T) or (1 x T) - list, array, or tensor of ints.  
              These are characters that have been encoded to integers.
            - embedding_weights (float) - (epsilon) The magnitude of embedding for vowel/consonents
        
        Return: 
            - An encoding that encodes whether each character is a vowel or consonant.  For spaces, we give an embedding of zero
    """
    pos_embed = torch.zeros_like(context, dtype=torch.float32)
    pos_embed += embedding_weight * sum(context==i for i in vowelidxs)
    pos_embed -= embedding_weight * sum(context==i for i in consonantidxs)
    return pos_embed

def plot_logs(final_bpc, loss_logs): 

    f = open(loss_logs)
    losses, bpcs = [], []
    for line in f: 
        losses.append(ast.literal_eval(line)[0])
        bpcs.append(ast.literal_eval(line)[1])
        

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f"Final bpc of {final_bpc:.2f}")
    fig.text(0.5, 0.04, 'iter', ha='center')
    axs[0].plot(range(len(losses)), losses)
    axs[0].set(ylabel="Loss")
    axs[0].grid()

    axs[1].plot(range(len(bpcs)), bpcs)
    axs[1].set(ylabel='bpc')
    axs[0].grid()
    plt.savefig(loss_logs.replace(loss_logs.split('/')[-1], "losses.png"))
    fig.show()

@torch.no_grad()
def validation_bpc(data_dir:str, block_size:int, model:nn.Module, softmax:nn.Module, 
                   device:torch.device="cuda", ctx:torch.amp.autocast_mode.autocast=None,
                   future_window:int=1):
    """This goes through the entire validation set, calculating the probability of the next character
    for each block, and finally calculates a final bpc after training. 

    Args: 
        - data_dir (str, file-like) the directory of training data
        - block_size (int) - context window.  Number of characters used in context
        - model (nn.module) - the GPT model defined in model.py and created in train.py 
        - softmax (nn.module) - simple softmax layer to collect probabilities from logits
        - device (torch.device) - 'cuda' or 'cpu' 
        - ctx (torch.autocast) - pass this from train.py 
        - future_window (int)  - how many characters we are trying to estimate in the future
    """
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    data = data[len(data) // 2 : len(data) // 2 + 20000]

    # Number of characters we are estimating.. 
    T = len(data)-block_size
    bpc = 0
    for i in tqdm(range(T)):    
        x = torch.from_numpy((data[i:i+block_size]).astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)).unsqueeze(0)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        with ctx:
                logits, loss = model(x, y)

        softmaxed = softmax(logits[:,-1,:])
        prob_truechar = [np.log2(softmaxed[i][y[i,-future_window]].cpu().detach().to(dtype=torch.float16).numpy()) 
                            for i in range(softmaxed.shape[0])]
        
        bpc -= sum(prob_truechar)
    
    bpc /= T
    print(f"Final bpc over validation set: {bpc}")
    return bpc

@torch.no_grad()
def convolve(A, V):
    """Performs convolution.
     Each row vector in V performs convolution on the corresponding row
     in A. 

     Args:
        - A (BS, sequence_length), a matrix of encoded characters
        - V (BS, kernel_size), matrix where each row is a convolution kernel
    """
    BS, T = A.shape
    A = A.unsqueeze(0)
    V = V.flip(1).unsqueeze(1)
    out = torch.conv1d(A, V, groups = BS).squeeze()
    return out
