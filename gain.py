import torch
import numpy as np
# from tqdm import tqdm
from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F
from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F

dataset_file = 'Spam.csv'  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
use_gpu = False  # set it to True to use GPU and False to use CPU

if use_gpu:
    torch.cuda.set_device(0)

#%% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

#################### DATA #######################
Data = np.loadtxt(dataset_file, delimiter=",",skiprows=1)

# Parameters
No = len(Data)
Dim = len(Data[0,:])

# Hidden state dimensions
H_Dim1 = H_Dim2 = Dim


#################### NORMALIZATION #######################

Min_Val = np.min(Data, axis=0)
Data -= Min_Val
Max_Val = np.max(Data, axis=0)
Data /= (Max_Val + 1e-6)
Data

#%% Missing introducing
p_miss_vec = p_miss * np.ones((Dim,1)) 
   
Missing = np.zeros((No,Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size = [len(Data),])
    B = A > p_miss_vec[i]
    Missing[:,i] = 1.*B

       
#################### TRAIN TEST DIVISION #######################

idx = np.random.permutation(No)

Train_No = int(No * train_rate)
Test_No = No - Train_No
    
# Train / Test Features
trainX = Data[idx[:Train_No],:]
testX = Data[idx[Train_No:],:]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_No],:]
testM = Missing[idx[Train_No:],:]


#################### NECESSARY FUNCTIONS #######################
# 1. Xavier Initialization Definition

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)
    
# Hint Vector Generation
def sample_M(m, n, p):
    return (np.random.uniform(0., 1., size=(m, n)) > p).astype(float)


#%% 1. Discriminator
if use_gpu is True:
    D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
    D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
    D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")       # Output is multi-variate
else:
    D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
    D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
    D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)       # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

#%% 2. Generator
if use_gpu is True:
    G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
    G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
    G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")
else:
    G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
    G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
    G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

#%% 1. Generator
def generator(new_x,m):
    inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
    
    return G_prob

#%% 2. Discriminator
def discriminator(new_x, h):
    inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
    
    return D_prob

#%% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def discriminator_loss(M, New_X, H):
    G_sample = generator(New_X, M)
    Hat_New_X = New_X * M + G_sample * (1 - M)
    D_prob = discriminator(Hat_New_X, H)

    # Adversarial loss for discriminator
    D_loss = -torch.mean(M * torch.log(torch.clamp(D_prob, 1e-8, 1.0)) + (1 - M) * torch.log(torch.clamp(1 - D_prob, 1e-8, 1.0)))
    return D_loss

def generator_loss(X, M, New_X, H, alpha=10):
    G_sample = generator(New_X, M)
    Hat_New_X = New_X * M + G_sample * (1 - M)
    D_prob = discriminator(Hat_New_X, H)

    # Adversarial loss for generator
    G_loss1 = -torch.mean((1 - M) * torch.log(torch.clamp(D_prob, 1e-8, 1.0)))

    # MSE loss for generator
    MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss

    # MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample)**2) / torch.mean(1 - M)
    return G_loss, MSE_train_loss, MSE_test_loss

def test_loss(X, M, New_X):
    G_sample = generator(New_X, M)

    # MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample)**2) / torch.mean(1 - M)
    return MSE_test_loss, G_sample


optimizer_D = torch.optim.Adam(params=theta_D)
optimizer_G = torch.optim.Adam(params=theta_G)


