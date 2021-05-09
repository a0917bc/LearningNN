#%%
'''
This code is came from the following website
https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f38
'''
import numpy as np
import cv2
def processImage(image): 
  image = cv2.imread(image) 
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
  return image

def paddingImage(image, padding=0):
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    return imagePadded
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
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
        
        print(imagePadded)# Print 3S
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
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
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
#%%
#The driver code for conv2d is to do edge-detection
# if __name__ == '__main__':
#     # Grayscale Image
#     image = processImage('Image.jpeg')

#     # Edge Detection Kernel
#     #kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
#     kernel = np.ones((3, 3))
#     kernel = kernel / 9
    

#     # Convolve and Save Output
#     #output = convolve2D(image, kernel, padding=2)
#     smoothed = convolve2D(image, kernel, padding=0)
#     smoothed = paddingImage(smoothed, 1)
#     detail = image - smoothed
#     sharpened = image + detail
#     cv2.imwrite('sharpenedImage.jpg', sharpened)
#     #cv2.imwrite('2DConvolved.jpg', output)
#%%
if __name__ == '__main__':
    image = processImage('Image.jpeg')
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # imagePadded = paddingImage(image, padding=0)
    output = convolve2D(image, kernel, padding=1)
    # cv2.imwrite('2DConvolved.jpg', output)
    #print(output)
#%%
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
A = np.arange(9).reshape(3, 3)
A1 = paddingImage(A, padding=1)
A2 = convolve2D(A1, kernel)
#%%

# %%
