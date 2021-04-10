#%%
import math


a1_1 = 0.5
a2_1 = 0.5
ans = 1
b1_2 = 0
b2_2 = 0
b1_3 = 0
w11_2 = 0.5
w12_2 = 0.5
w21_2 = 0.5
w22_2 = 0.5
w11_3 = 0.5
w12_3 = 0.5

partialJpartiala1_3 = 0
partialJpartialb1_3 = 0
partialJpartialw11_3 = 0
partialJpartialw12_3 = 0
partialJpartiala1_2 = 0
partialJpartiala2_2 = 0

partialJpartialb1_2 = 0
partialJpartialb2_2 = 0
partialJpartialw11_2 = 0
partialJpartialw12_2 = 0
partialJpartialw21_2 = 0
partialJpartialw22_2 = 0
learningRate = 10
absJ = 0.0
Count = 0
#partialJpartiala1_3 = 0
#partialJpartiala1_3 = 0
#partialJpartiala1_3 = 0
running = True
Tolerance = 0.001
Error = 0
while running:
#Evaluate
    Count += 1
    print(Count)
    z1_2 = w11_2 * a1_1 + w12_2 * a2_1 + b1_2
    a1_2 = 1 / (1 + math.exp(-z1_2)) 
    z2_2 = w21_2 * a1_1 + w22_2 * a2_1 + b2_2
    a2_2 = 1 / (1 + math.exp(-z2_2)) 
    z1_3 = w11_3 * a1_2 + w12_3 * a2_2 + b1_3
    a1_3 = 1 / (1 + math.exp(-z1_3)) 
    print('Inference = ' + str(a1_3))
    J = (-1/2) * (ans - a1_3)**2#Loss function
    #print(J)
    #absJ = abs(J)
    Error = ans - a1_3
    print('Error = ' + str(Error))
    if Error < Tolerance:
        print('Successful!')
        running = False
#BackwardPropagation
    partialJpartiala1_3 = -(ans - a1_3)

    partialJpartialb1_3 = a1_3 * (1 - a1_3) * partialJpartiala1_3
    partialJpartialw11_3 = a1_2 * partialJpartialb1_3 
    partialJpartialw12_3 = a2_2 * partialJpartialb1_3
    partialJpartiala1_2 = w11_3 * partialJpartialb1_3
    partialJpartiala2_2 = w12_3 * partialJpartialb1_3

    partialJpartialb1_2 = a1_2 * (1 - a1_2) * partialJpartiala1_2
    partialJpartialb2_2 = a2_2 * (1 - a2_2) * partialJpartiala2_2
    partialJpartialw11_2 = a1_1 * partialJpartialb1_2
    partialJpartialw12_2 = a2_1 * partialJpartialb1_2
    partialJpartialw21_2 = a1_1 * partialJpartialb2_2
    partialJpartialw22_2 = a2_1 * partialJpartialb2_2

#Update
    b1_2 -= learningRate * partialJpartialb1_2 
    b2_2 -= learningRate * partialJpartialb2_2 
    b1_3 -= learningRate * partialJpartialb1_3 
    w11_2 -= learningRate * partialJpartialw11_2 
    w12_2 -= learningRate * partialJpartialw12_2 
    w21_2 -= learningRate * partialJpartialw21_2 
    w22_2 -= learningRate * partialJpartialw22_2 
    w11_3 -= learningRate * partialJpartialw11_3
    w12_3 -= learningRate * partialJpartialw12_3

# %%
