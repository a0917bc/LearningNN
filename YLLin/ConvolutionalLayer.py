#%%
import numpy as np
import cv2
def processImage(image):
  image = cv2.imread(image)
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  return image

def paddingImage(image, padding=0):
    if padding != 0:
        imagePadded = np.zeros(
            (image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding),
                    int(padding):int(-1 * padding)] = image
    return imagePadded


def convolve2D(image, kernel, padding=0, flag_bp=False, strides=1):
    # Cross Correlation
    
    if(flag_bp):
        kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        # imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image

        imagePadded = paddingImage(image, padding)

        # print(imagePadded)  # Print 3S
    else:
        imagePadded = image

    # Iterate through imagePadded
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        # numpy have dot operator so we don't need to another two for loop
                        output[x, y] = (
                            kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    # for y in range(image.shape[1]):
    #     # Exit Convolution
    #     # Only Convolve if y has gone down by the specified Strides
    #     if y % strides == 0:
    #         for x in range(image.shape[0]):
    #             # Go to next row once kernel is out of bounds
    #             if x % strides == 0:
    #                     # numpy have dot operator so we don't need to another two for loop
    #                     output[x, y] = (kernel * imagePadded[x:x +
    #                      xKernShape, y: y + yKernShape]).sum()

    return output
# Just Conv2d operation, need to extend the channel form for CNN
def convolutional_layer():
    for r in range(H_o):
        for c in range(W_o):
            for ch in range(C_o):
                for i in range(H_k):
                    for j in range(W_k):
                        for k in range(C_k):   
                            zz[ch] = conkernel[k, ch]*a[k]  


def convolutional_layer(input_feature_map, kernel, padding=0, flag_bp=False):
    C_I = input_feature_map.shape[0]
    if(flag_bp):
        kernel = kernel.reshape((int(kernel.shape[1]), int(kernel.shape[0]), int(kernel.shape[2]), int(kernel.shape[3])))
    C_O = kernel.shape[1]
    H_K = len(kernel[0, 0, :, 0])
    W_K = len(kernel[0, 0, 0, :])
    H_I = len(input_feature_map[0, :, 0])
    W_I = len(input_feature_map[0, 0, :])
    output_feature_map = np.zeros(
        (C_O, H_I-H_K+1+padding*2, W_I-W_K+1+padding*2))
    for cout in range(C_O):
        for cin in range(C_I):
            tmp = convolve2D(
                input_feature_map[cin], kernel[cin, cout], padding, flag_bp)
            # if cout == 8:
            #     print((tmp))
            output_feature_map[cout] += tmp
    return output_feature_map


def convolutional_layer_kernel_bp(dJdzz, a):
    C_I = a.shape[0]
    kernel_maps = dJdzz.shape[0]
    dJdk = np.zeros((C_I, kernel_maps, a.shape[1]-dJdzz.shape[1]+1, a.shape[2]-dJdzz.shape[2] + 1))
    for i in range(C_I):
        for j in range(kernel_maps):
            dJdk[i][j] = convolve2D(a[i], dJdzz[j])
    return dJdk


# a = np.random.randint(0, 19, size=(1, 5, 5))
# test_kernel1 = np.random.randint(0, 10, size=(10, 1, 3, 3))
# test_kernel = np.random.randint(0, 10, size=(1, 10, 3, 3))

# zz = convolutional_layer(a, test_kernel, 1, 10)
# dJda = convolutional_layer(zz, test_kernel1, 10, 1)


#%%
