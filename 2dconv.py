"""
Reference: 
    Alisaaalehi. “Alisaaalehi/convolution_as_multiplication.” GitHub, https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb.
"""
import numpy as np
from scipy.linalg import toeplitz
import time
import cv2
import matplotlib.pyplot as plt


def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    input = np.flipud(input)     # flipping the input matrix 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    output=np.flipud(output)     # flipping the output matrix 
    return output

def conv2dmatrix(I, H, print_ir=False):
    t = time.process_time()     #Start time
    I_row_num, I_col_num = I.shape 
    H_row_num, H_col_num = H.shape
    output_row_num = I_row_num + H_row_num - 1
    output_col_num = I_col_num + H_col_num - 1    
    H_zero_padded = np.pad(H, ((output_row_num - H_row_num, 0),
                               (0, output_col_num - H_col_num)),
                            'constant', constant_values=0) # zero padding the filter
    
    toeplitz_list = []
    for i in range(H_zero_padded.shape[0]-1, -1, -1): 
        c = H_zero_padded[i, :] 
        r = np.r_[c[0], np.zeros(I_col_num-1)] 
        toeplitz_m = toeplitz(c,r) 
        toeplitz_list.append(toeplitz_m)
    c = range(1, H_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)
    b_h, b_w = toeplitz_shape # height and widths of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
    vectorized_I = matrix_to_vector(I)     # convert I to a vector
    result_vector = np.matmul(doubly_blocked, vectorized_I)
    out_shape = [output_row_num, output_col_num]     # reshape the raw rsult to desired matrix form
    output = vector_to_matrix(result_vector, out_shape)
    elapsed_time = time.process_time() - t
    from scipy import signal
    signallib_op = signal.convolve2d(I, H, "full")
    print('Result is\n', signallib_op)
    error = signallib_op - output     #Caluculating error by comparing 
    print('Error is \n',error)
    print('Time taken is ',elapsed_time)
    return output, elapsed_time, error

I = np.arange(1,10).reshape(3,3)
H = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
Conv2d = conv2dmatrix(I, H)

elephant_gray = cv2.imread("elephant1.jpeg", 0)
#elephant = cv2.resize(elephant_gray, (200,200))
elephant=elephant_gray
print('Result for the elephant image');
output = conv2dmatrix(elephant, H)
plt.figure()
plt.imshow(output[0], cmap = 'gray')
plt.title(" Matrix multiplication 2-D Convolution")
plt.show()