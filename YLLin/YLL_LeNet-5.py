#%%
from ConvolutionalLayer import *
from MaxPoolingLayer import *
from MLP import *
from mnist import *
import numpy as np
Layers = [(6, 28, 28), (6, 14, 14), (16, 10, 10), (16, 5, 5)]

c1_zz = np.zeros((6, 28, 28), dtype='float16')
c1_dJdzz = c1_zz
c1_a = np.zeros((6, 28, 28), dtype='float16')
c1_dJda = c1_a

s2_z = np.zeros((6, 14, 14), dtype='float16')
s2_dJdz = s2_z
s2_dJdzz_filter = s2_dJdz

c3_zz = np.zeros((16, 10, 10), dtype='float16')
c3_dJdzz = c3_zz
c3_a = np.zeros((16, 10, 10), dtype='float16')
c3_dJda = c3_a

s4_z = np.zeros((16, 5, 5), dtype='float16')
s4_dJdz = s4_z
s4_dJdzz_filter = s4_dJdz
s4_z_flatten = np.zeros((s4_z.size, 1), dtype='float16')

c1_kernel = np.random.randn(1, 6, 5, 5)
c1_dJdk = c1_kernel
c1_b = np.random.randn(6, 1, 1)
c1_dJdb = c1_b
c3_kernel = np.random.randn(6, 16, 5, 5)
c3_dJdk = c3_kernel
c3_b = np.random.randn(16, 1, 1)
c3_dJdb = c3_b

nn = MLP(Layers=(400, 120, 84, 10), BatchSize=1)
nn.lr = 0.001
nn.act = 'tanh'
nn.bs = 1

training_data = list(read(dataset='training'))
testing_data = list(read(dataset='testing'))


# function = MLP()
for case in range(len(training_data)):
    if (case==100):
        break
    a0 = np.ones((32, 32), dtype='float16') * -1
    prep_data = training_data[case][1].astype(float)
    label = training_data[case][0]
    label = np.array(label).reshape(1, )
    print(case, label)
    prep_data[prep_data > 0.0] = 1.175
    a0[2:-2, 2:-2] = prep_data
    a0 = a0.reshape(1, 32, 32)
    c1_zz = convolutional_layer(a0, c1_kernel) + c1_b
    c1_a = nn.tanh(c1_zz)
    s2_z, s2_dJdzz_filter = poolingLayer(c1_a)
    c3_zz = convolutional_layer(s2_z, c3_kernel) + c3_b
    c3_a = nn.tanh(c3_zz)
    s4_z, s4_dJdzz_filter = poolingLayer(c3_a)
    s4_z_flatten = s4_z.flatten().reshape(s4_z.size, 1)
    nn.forward(s4_z_flatten)

    #loss
    nn.loss(label)
    nn.backprop()
    nn.update()

    # backprop in CNN
    s4_dJdz = nn.net[0]['a'].reshape(s4_z.shape)  # (16, 5, 5)
    # bp from s4 to c3; (16, 10, 10)
    c3_dJda = poolingLayer_bp(s4_dJdz, s4_dJdzz_filter)
    c3_dJdzz = nn.tanhPrime(c3_a) * c3_dJda  # find dJdzz = dJda * dadzz
    c3_dJdk = convolutional_layer_kernel_bp(c3_dJdzz, s2_z)  # (16, 10, 10) * (6, 14, 14)
    c3_dJdb = c3_dJdzz.sum(axis=0)
    # bp from c3 to s2
    s2_dJdz = convolutional_layer(c3_dJdzz, c3_kernel, padding=4, flag_bp=True)
    # bp from s2 to c1; (6, 28, 28)
    c1_dJda = poolingLayer_bp(s2_dJdz, s2_dJdzz_filter)
    c1_dJdzz = nn.tanhPrime(c1_a) * c1_dJda
    c1_dJdk = convolutional_layer_kernel_bp(c1_dJdzz, a0)
    c1_dJdb = c1_dJdzz.sum(axis=0)

    # update in CNN
    c1_kernel = c1_kernel - nn.lr * c1_dJdk
    c1_b = c1_b - nn.lr * c1_dJdb
    c3_kernel = c3_kernel - nn.lr * c3_dJdk
    c3_b = c3_b - nn.lr * c3_dJdb

correct_num = 0
correct_rate = float()
for case in range(len(testing_data)):
    a0 = np.ones((32, 32), dtype='float16') * -1
    prep_data = testing_data[case][1].astype(float)
    label = testing_data[case][0]
    label = np.array(label).reshape(1, )
    print(case, label)
    prep_data[prep_data > 0.0] = 1.175
    a0[2:-2, 2:-2] = prep_data
    a0 = a0.reshape(1, 32, 32)
    c1_zz = convolutional_layer(a0, c1_kernel) + c1_b
    c1_a = nn.tanh(c1_zz)
    s2_z, s2_dJdzz_filter = poolingLayer(c1_a)
    c3_zz = convolutional_layer(s2_z, c3_kernel) + c3_b
    c3_a = nn.tanh(c3_zz)
    s4_z, s4_dJdzz_filter = poolingLayer(c3_a)
    s4_z_flatten = s4_z.flatten().reshape(s4_z.size, 1)
    nn.forward(s4_z_flatten)
    if(nn.yhat==label):
        correct_num +=1
    correct_rate = correct_num / float(testcase_num)
print("Testing result is" + str(correct_rate))

# %%
