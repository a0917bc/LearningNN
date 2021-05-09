#%%
import numpy as np
import math
import matplotlib.pyplot as plt
#%matplotlib inline, already solved by VS code
np.set_printoptions(precision = 2)
class MLP(object):
    '''Multi_Layer Perceptron, Fully-Connected Layer'''
    
    def __init__(self, Layers = (2, 2, 3), BatchSize = 4):
        self.bs = BatchSize
        self.lr = float()
        self.LeakyRate = float()
        self.act = str()
        self.net = [dict() for _ in range(len(Layers))]# Every element in the list is a dictionary 

        self.net[0]['a'] = np.zeros((Layers[0], self.bs), dtype = 'float16')
        self.net[0]['dJda'] = np.zeros(self.net[0]['a'].shape, dtype = 'float16')
        
        for i in range(1, len(Layers)):
            self.net[i]['a'] = np.zeros((Layers[i], self.bs), dtype = 'float16')
            self.net[i]['z'] = np.zeros(self.net[i]['a'].shape, dtype = 'float16')
            self.net[i]['W'] = np.random.randn(Layers[i], Layers[i - 1]).astype('float16')
            self.net[i]['b'] = np.random.randn(Layers[i], 1).astype('float16')
            self.net[i]['dJda'] = np.zeros(self.net[i]['a'].shape, dtype = 'float16')
            self.net[i]['dJdz'] = np.zeros(self.net[i]['z'].shape, dtype = 'float16')
            self.net[i]['dJdW'] = np.zeros(self.net[i]['W'].shape, dtype = 'float16')
            self.net[i]['dJdb'] = np.zeros(self.net[i]['b'].shape, dtype = 'float16')
        
        self.p = np.zeros(self.net[-1]['a'].shape, dtype = 'float16') # Softmax Out
        self.dJdp = np.zeros(self.p.shape, dtype='float16')

        self.yhat = np.zeros(self.bs, dtype=int) # Predicted Answer
        # self.yhat_onehot = np.zeros(self.p.shape, dtype=int)
        self.y_onehot = np.zeros(self.p.shape, dtype=int)

        self.J = []
        self.W_trace = []


    def softmax(self, a):
        delta = 1.0
        return np.exp(a + delta) / np.sum(np.exp(a + delta), axis=0)
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-1.0 * z))
    def sigmoidPrime(self, a): # sigmoid's derivative
        return a * (1.0 -a)
    def tanh(self, z):
        return np.tanh(z)
    def tanhPrime(self, a):
        return 1 - pow(np.tanh(a), 2)
    def ReLU(self, z):
        a = np.copy(z)
        a[a < 0] = 0.0
        return a
    def ReLUPrime(self, a):
        dadz = np.copy(a)
        dadz[a > 0] = 1.0
        return dadz
    def activation(self, z):
        a = np.copy(z)
        if self.act == 'sigmoid':
            a = self.sigmoid(z)
        elif self.act == 'ReLU':
            a = self.ReLU(z)
        elif self.act == 'tanh':
            a = self.tanh(z)
        else:
            print("Activation Selection Error")
        return a
    def activationPrime(self, a):
        if self.act == 'sigmoid':
            z = self.sigmoidPrime(a)
        elif self.act == 'ReLU':
            z = self.ReLUPrime(a)
        elif self.act == 'tanh':
            z = self.tanhPrime(a)
        else:
            print("Activation Selection Error")
        return z
    # Warren
    def forward(self, x): # "copyto" looks like directly manipulate the value at some address
        np.copyto(self.net[0]['a'], x) # Copy input into a of Layer 0

        for i in range(1, len(self.net)): # Start from 0; Let i start from 1
            np.copyto(self.net[i]['z'], np.dot(self.net[i]['W'], self.net[i-1]['a']) + self.net[i]['b']) # iterator
            np.copyto(self.net[i]['a'], self.activation(self.net[i]['z']))
        
        np.copyto(self.p, self.softmax(self.net[-1]['a']))
        np.copyto(self.yhat, np.argmax(self.p, axis=0))# Final predicated answer
        return

    def loss(self, y):
        '''only a "1" '''
        self.y_onehot.fill(0)
        for i in range(self.bs):
            self.y_onehot[y[i], i] = 1 # (Compare & Turn on????)
        loss_value = -1.0 * np.sum(self.y_onehot * np.log(self.p)) / self.bs

        self.J.append(loss_value) # Look 
# fffff
        # W = []
        # for i in range(1, len(self.net)):
        #     W.extend(np.ravel(self.net[i]['W'])) # ravel =? extend
        #     W.extend(np.ravel(self.net[i]['b']))
        # self.W_trace.append(W)
        # return

    def backprop(self):
        self.dJdp = 1.0 / (1.0 - self.y_onehot - self.p)

        dpda = np.array([[self.p[i, :] * (1.0 - self.p[j, :]) if i == j
                          else -1 * self.p[i, :] * self.p[j, :]
                          for i in range(self.p.shape[0])]
                          for j in range(self.p.shape[0])])
        for i in range(self.bs):
            self.net[-1]['dJda'][:, i] = np.dot(dpda[:, :, i], self.dJdp[:, i])

        for i in range((len(self.net) - 1), 0, -1):# computed function 
            np.copyto(self.net[i]['dJdz'], (self.net[i]['dJda'] * self.activationPrime(self.net[i]['a']))) # dJdz = dJda * dadz

            np.copyto(self.net[i]['dJdb'], np.mean(self.net[i]['dJdz'], axis = 1)[:, None]) #dJdb = mean(dJdz.sum())

            np.copyto(self.net[i]['dJdW'], np.dot(self.net[i]['dJdz'], self.net[i-1]['a'].T) / self.bs) # dJdw = dJdz x (a^(i-1))^T

            np.copyto(self.net[i - 1]['dJda'], np.dot((self.net[i]['W']).T, self.net[i]['dJdz'])) # dJda^(i-1) = (W^i)^T x dJdz^i
        return

    def update(self):
        for i in range(1, len(self.net)):
            np.copyto(self.net[i]['W'], self.net[i]['W'] - self.lr * self.net[i]['dJdW']) # += may work
            #self.net[i]['W'] -= self.lr * self.net[i]['dJdW']
            np.copyto(self.net[i]['b'], self.net[i]['b'] - self.lr * self.net[i]['dJdb'])
        return
    
    def train(self, train_x, train_y, epoch_count = 1000, lr = 0.01, act = "sigmoid", LeakyRate = 0.1):
        self.lr = lr
        self.act = act
        self.LeakyRate = LeakyRate

        for _ in range(epoch_count): # constant times for, solved:how to eliminate warning the variable isn't used 
            for i in range(train_x.shape[1] // self.bs): 
                x = train_x[:, i * self.bs : (i + 1) * self.bs]
                y = train_y[i * self.bs : (i + 1) * self.bs]
                self.forward(x)
                self.loss(y)
                self.backprop()
                self.update()
        return

    def inference(self, inference_x):
        yhat = []
        for i in range(inference_x.shape[1]//self.bs):
            x = inference_x[:, i * self.bs:(i + 1) * self.bs]
            self.forward(x)
            yhat.extend(list(self.yhat))
        return yhat
    
    def plot_W(self):
        curve = [[] for i in range(len(self.W_trace[0]))]
        for j in range(len(self.W_trace[0])):
            for i in range(len(self.W_trace)):
                curve[j].append(self.W_trace[i][j])
        for c in curve:
                plt.plot(c)

        return


# %%

#train_x = np.array([[1, 3, 3, 1, 2, 2, 1, 3], [1, 2, 4, 2, 1, 3, 3, 4]])
#train_y = np.array([0, 1, 2, 0, 0, 1, 1, 2])
# tx = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6], [1, 2, 3, 1, 2, 4, 5, 1, 3, 4, 5, 1, 3, 5, 6, 3, 4, 4]])
# ty = np.array([1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 1])

# nn = MLP((2, 5, 5, 3), 18)
# nn.train(tx, ty, epoch_count=50000, lr=0.01, act="sigmoid", LeakyRate=0.01)

# plt.plot(nn.J)
#%%
# import numpy as np
# Layers = (2, 5, 5, 3)
# BatchSize = 4
# lr = float()
# LeakyRate = float()
# act = str()
# net = [dict() for i in range(len(Layers))]

# net[0]['a'] = np.zeros((Layers[0], BatchSize), dtype = 'float16')
# net[0]['dJda'] = np.zeros(net[0]['a'].shape, dtype = 'float16')
        
# for i in range(1, len(Layers)):
#     net[i]['a'] = np.zeros((Layers[i], BatchSize), dtype = 'float16')
#     net[i]['z'] = np.zeros(net[i]['a'].shape, dtype = 'float16')
#     net[i]['W'] = np.random.randn(Layers[i], Layers[i - 1]).astype('float16')
#     net[i]['b'] = np.random.randn(Layers[i], 1).astype('float16')
#     net[i]['dJda'] = np.zeros(net[i]['a'].shape, dtype = 'float16')
#     net[i]['dJdz'] = np.zeros(net[i]['z'].shape, dtype = 'float16')
#     net[i]['dJdW'] = np.zeros(net[i]['W'].shape, dtype = 'float16')
#     net[i]['dJdb'] = np.zeros(net[i]['b'].shape, dtype = 'float16')
        
# p = np.zeros(net[-1]['a'].shape, dtype = 'float16') # Softmax Out
# dJdp = np.zeros(p.shape, dtype='float16')

# yhat = np.zeros(BatchSize, dtype=int) # Predicted Answer
# y_onehot = np.zeros(p.shape, dtype=int)

# J = []
# W_trace = []
# %%
# 怎麼用::-1來reverse每一個Matrix
# Why loss J find minimum is 15 
    # Hand calculate for it.
'''
x3
array([[[8, 1, 5, 9, 8],
        [9, 4, 3, 0, 3],
        [5, 0, 2, 3, 8],
        [1, 3, 3, 2, 7]],

       [[0, 1, 9, 9, 0],
        [4, 7, 3, 2, 7],
        [2, 0, 0, 4, 5],
        [5, 6, 8, 2, 1]],

       [[4, 9, 8, 1, 1],
        [7, 9, 9, 3, 6],
        [7, 2, 0, 3, 5],
        [9, 4, 4, 2, 4]]])

x3[0, ::-1] all elements(each row) are reversed
array([[1, 3, 3, 2, 7],
       [5, 0, 2, 3, 8],
       [9, 4, 3, 0, 3],
       [8, 1, 5, 9, 8]])
'''
#%%
'''
Sexy classifier
'''
# px = np.array([[1, 1, 2, 2, 2, 4, 4, 5, 5], [2, 5, 1, 2, 4, 1, 4, 2, 4]])
# py = np.array([0, 2, 0, 0, 2, 1, 3, 1, 3])
# nnp = MLP((2, 6, 4), 9)
# nnp.train(px, py, epoch_count=10000, lr=0.05, act="sigmoid", LeakyRate=0)
# plt.plot(nnp.J)
# print(nnp.yhat)
# print(py)
# %%
