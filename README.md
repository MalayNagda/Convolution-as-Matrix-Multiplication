# Convolution as Matrix Multiplication

Perfomred convolution on an elephant image as a matrix multiplication using a vertical edge filter matrix. The reference used for this purpose is- https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb.

The vertical edge filter matrix-

| 1 | 0 | -1 |
| ------------- | ------------- | ------------- | 
| **1** | **0** | **-1** |
| **1** | **0** | **-1** |

The original elephant image is-
<p align="center">
<img src="images/elephant1.jpeg">
</p>

The edge filtered image after 2d convolution is-
<p align="center">
<img src="images/2dconv.png">
</p>