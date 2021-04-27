#%%
from numpy import *
from ConvNet import *
import time
import struct

def train_net(train_covnet, logfile, cycle, learn_rate, case_num = -1) :
    # Read data 
    # Change it to your own dataset path
    trainim_filepath = 'MNIST/train-images.idx3-ubyte'
    trainlabel_filepath = 'MNIST/train-labels.idx1-ubyte'
    trainimfile = open(trainim_filepath, 'rb')
    trainlabelfile = open(trainlabel_filepath, 'rb')
    train_im = trainimfile.read()
    train_label = trainlabelfile.read()
    im_index = 0
    label_index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , train_im , im_index)
    magic, numLabels = struct.unpack_from('>II', train_label, label_index)
    print ('train_set:', numImages)

    train_btime = time.time()
    logfile.write('learn_rate:' + str(learn_rate) + '\n')
    logfile.write('train_cycle:' + str(cycle) + '\n')

    # Begin to train
    for c in range(cycle) :
        im_index = struct.calcsize('>IIII')
        label_index = struct.calcsize('>II')
        train_case_num = numImages
        if case_num != -1 and case_num < numImages :
            train_case_num = case_num
        logfile.write("trainset_num:" + str(train_case_num) + '\n')
        for case in range(train_case_num) :
            im = struct.unpack_from('>784B', train_im, im_index)
            label = struct.unpack_from('>1B', train_label, label_index)
            im_index += struct.calcsize('>784B')
            label_index += struct.calcsize('>1B')
            im = array(im)
            im = im.reshape(28,28)
            bigim = list(ones((32, 32)) * -0.1)
            for i in range(28) :
                for j in range(28) :
                    if im[i][j] > 0 :
                        bigim[i+2][j+2] = 1.175
            im = array([bigim])
            label = label[0]
            print (case, label)
            train_covnet.fw_prop(im, label)
            train_covnet.bw_prop(im, label, learn_rate[c])

    print ('train_time:', time.time() - train_btime)
    logfile.write('train_time:'+ str(time.time() - train_btime) + '\n')
def test_net(train_covnet, logfile, case_num = -1) :
    
    # Read data 
    # Change it to your own dataset path
    testim_filepath = 'MNIST/t10k-images.idx3-ubyte'
    testlabel_filepath = 'MNIST/t10k-labels.idx1-ubyte'
    testimfile = open(testim_filepath, 'rb')
    testlabelfile = open(testlabel_filepath, 'rb')
    test_im = testimfile.read()
    test_label = testlabelfile.read()

    im_index = 0
    label_index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , test_im , im_index)
    magic, numLabels = struct.unpack_from('>II', test_label, label_index)
    print('test_set:', numImages)
    im_index += struct.calcsize('>IIII')
    label_index += struct.calcsize('>II')
    
    correct_num = 0
    testcase_num = numImages
    if case_num != -1 and case_num < numImages:
        testcase_num = case_num
    logfile.write("testset_num:" + str(testcase_num) + '\n')

    # To test
    for case in range(testcase_num) :
        im = struct.unpack_from('>784B', test_im, im_index)
        label = struct.unpack_from('>1B', test_label, label_index)
        im_index += struct.calcsize('>784B')
        label_index += struct.calcsize('>1B')
        im = array(im)
        im = im.reshape(28,28)
        bigim = list(ones((32, 32)) * -0.1)
        for i in range(28) :
            for j in range(28) :
                if im[i][j] > 0 :
                    bigim[i+2][j+2] = 1.175
        im = array([bigim])
        label = label[0]
        print(case, label)
        train_covnet.fw_prop(im)
        if argmax(train_covnet.outputlay7.maps[0][0]) == label :
            correct_num += 1
    correct_rate = correct_num / float(testcase_num)
    print('test_correct_rate:', correct_rate)
    logfile.write('test_correct_rate:'+ str(correct_rate) + '\n')
    logfile.write('\n')
#%%
'''
This code is a lot of shit
'''
log_timeflag = time.time()
train_covnet = CovNet()
# Creat a folder name 'log' to save the history
train_covnet.print_netweight('log/origin_weight' + str(log_timeflag) + '.log')
logfile = open('log/nanerrortestcase.log', 'w')
logfile.write("train_time:" + str(log_timeflag) + '\n')
train_net(train_covnet, logfile, 1, [0.01, 0.01], 10000)
train_covnet.print_netweight('log/trained_weight' + str(log_timeflag) + '.log')
test_net(train_covnet, logfile, 3000)
logfile.write('\n')
logfile.close()