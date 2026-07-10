######################################### 
### play-around the data
######################################### 

words = open('names.txt', 'r').read().splitlines()

words[:10]

len(words)

min(len(w) for w in words)

max(len(w) for w in words)

for w in words[:3]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)
  
#########################################    
### bigram
######################################### 

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
       
sorted(b.items(), key = lambda kv: -kv[1])

import torch

a = torch.zeros((3, 5), dtype=torch.int32)
a

a[1, 3] += 1
a

a[0,0] =  5
a

N = torch.zeros((27, 27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 
        
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
plt.imshow(N, cmap = 'Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color='gray')
plt.axis('off');


######################################### 
### sampling
######################################### 

p = N[0].float()
p = p/p.sum()
p

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples = 1, replacement=True, generator=g).item()
itos[ix]

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p/p.sum()
p

torch.multinomial(p, num_samples=100, replacement=True, generator=g)


######################################### 
### Probability Matrix and sampling
######################################### 

P = N.float()
P /= P.sum(1, keepdim=True)


g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    
    out = []
    ix = 0
    while True:
        
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


######################################### 
### loss function
######################################### 

"""
GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
equivalent to maximizing the log likelihood (because log is monotonic)
equivalent to minimizing the negative log likelihood
equivalent to minimizing the average negative log likelihood
"""

log_likelihood = 0.0
n = 0

for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1 
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')        
        
### .e: 0.0478 -3.0408
### em: 0.0377 -3.2793
### mm: 0.0253 -3.6772
### ma: 0.3899 -0.9418
### a.: 0.1960 -1.6299
### .o: 0.0123 -4.3982
### ol: 0.0780 -2.5508
### li: 0.1777 -1.7278
### iv: 0.0152 -4.1867
### vi: 0.3541 -1.0383
### ia: 0.1381 -1.9796
### a.: 0.1960 -1.6299
###.a: 0.1377 -1.9829
### av: 0.0246 -3.7045
### va: 0.2495 -1.3882
### a.: 0.1960 -1.6299
### log_likelihood=tensor(-38.7856)
### nll=tensor(38.7856)
### 2.424102306365967


# lossfunction of total set
        
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1 
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')      
        
### log_likelihood=tensor(-559891.7500)
### nll=tensor(559891.7500)
### 2.454094171524048


# test for any particular word

log_likelihood = 0.0
n = 0

#for w in words:
for w in ["rupa"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1 
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') 

### .r: 0.0512 -2.9727
### ru: 0.0198 -3.9199
### up: 0.0051 -5.2778
### pa: 0.2037 -1.5911
### a.: 0.1960 -1.6299
### log_likelihood=tensor(-15.3914)
### nll=tensor(15.3914)
### 3.07827091217041













