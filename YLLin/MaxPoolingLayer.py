import numpy as np
# def pooling(image, pooling_size, W_O, H_O):
#     index = np.zeros((W_O, H_O), dtype=int)
#     output = np.zeros((W_O, H_O))
#     dJdzz = np.zeros(image.shape)
#     for y in range(0, image.shape[1], pooling_size):
#         for x in range(0, image.shape[0], pooling_size):
#             index[x//2, y//2] = np.argmax(image[x:x+pooling_size, y:y+pooling_size])
#             output[x//2, y//2] = (image[x:x+pooling_size, y:y+pooling_size]).max()
#     return output, index


def pooling(image, pooling_size, W_O, H_O):
    index = int()
    output = np.zeros((W_O, H_O))
    dJdzz_filter = np.zeros(image.shape)
    for y in range(0, H_O):
        for x in range(0, W_O):
            pooling_filter = np.zeros(pooling_size*pooling_size)
            index = np.argmax(image[x*pooling_size:x*pooling_size+pooling_size, y *
                              pooling_size:(y+1)*pooling_size]) 
                            #   + (x*W_O+y) * pooling_size*pooling_size
            # pooling_filter = pooling_filter.flatten()
            pooling_filter[index] = 1
            pooling_filter = pooling_filter.reshape(pooling_size, pooling_size)
            output[x, y] = (image[x*pooling_size:x*pooling_size +
                            pooling_size, y*pooling_size:(y+1)*pooling_size]).max()
            dJdzz_filter[x*pooling_size:x*pooling_size +
                  pooling_size, y*pooling_size:(y+1)*pooling_size] = pooling_filter
    return output, dJdzz_filter

def poolingLayer_bp(dJdz, dJdzz_filter, pooling_size=2):
    channel = int(dJdz.shape[0])
    dJdzz = np.zeros(dJdzz_filter.shape, dtype='float16')
    dJdz_H_O = len(dJdz[0, 0, :])
    dJdz_W_O = len(dJdz[0, :, 0])
    for c in range(channel):
        for y in range(dJdz_H_O):
            for x in range(dJdz_W_O):
                dJdzz[c, x*pooling_size:(x+1)*pooling_size, y*pooling_size:(y+1)*pooling_size] = \
                dJdz[c, x, y] * dJdzz_filter[c, x*pooling_size:(x+1)*pooling_size, y*2:(y+1)*pooling_size]
    return dJdzz
# def pooling(image, pooling_size, W_O, H_O, I):

#     output = np.zeros((W_O, H_O))
#     for y in range(0, image.shape[1], pooling_size):
#         for x in range(0, image.shape[0], pooling_size):
#             output[x//2, y//2] = (image[x:x+pooling_size,
#                                   y:y+pooling_size]).mean()
#     return output


def poolingLayer(input_feature_map, pooling_size=2):   
    # Check whether it's 3D or not.
    W_O = input_feature_map.shape[1]//2
    H_O = input_feature_map.shape[2]//2
    C_I = input_feature_map.shape[0]
    dJdzz_filter = np.zeros(input_feature_map.shape)
    output_feature_map = np.zeros((C_I, W_O, H_O))
    for cin in range(C_I):
        output_feature_map[cin], dJdzz_filter[cin] = pooling(input_feature_map[cin], pooling_size, W_O, H_O)
    return output_feature_map, dJdzz_filter


# A = np.random.randint(0, 16, (3, 4, 4))
# poolingLayer(A, 3, 2)
