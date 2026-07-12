# -----------------------------------------------------------------------------
# import packages

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 


# -----------------------------------------------------------------------------
# read in all the words

words = open('names.txt', 'r').read().splitlines()
words[:8]
# output - ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']

len(words)
# output - 32033


# -----------------------------------------------------------------------------
# build the vocabulary of characters and mappings to/from integers

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)




