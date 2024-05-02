import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import rotate
from scipy import misc # contains an image of a racoon!
from PIL import Image # for reading image files

## make a one dimentional array -- a list
new_array = np.array([1.1, 9.2, 8.1, 4.7])
new_array.shape
## access the nth element -- 2nd
new_array[2]
## check the dimentions of the array
new_array.ndim

## build a 2 dimentional array
array_2d = np.array([[1, 2, 3, 9], 
                     [5, 6, 7, 8]])
## dimentions is a quantitative value of how many axies are in the matrix.
print(f"array_2d has the dimentions: {array_2d.ndim}")
print(f"the shape is: {array_2d.shape}")
print(f"it has {array_2d.shape[0]} rows and {array_2d.shape[1]} columns")
## access particular values in the matrix
array_2d[1, 2]
## access a particular row
array_2d[0, :] 

## build 3 dimentional array
mystery_array = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                        
                         [[7, 86, 6, 98],
                          [5, 1, 0, 4]],
                          
                          [[5, 36, 32, 48],
                           [97, 0, 27, 18]]])

mystery_array.ndim ## 3
mystery_array.shape ## 3, 2, 4 three pages of 2d arrays that are broken into two arrays that are 4 elements long
mystery_array[2, 1, 3] ## 18
mystery_array[2, 1, :] ## access the last vector in the array

## get the first element of each single array
mystery_array[:, :, 0] ## 0, 4, 7, 5, 5, 97

## build a list of ints starting at 10 and including everything up through 29, stepping by 1
a = np.arange(10, 30, 1)
print(a)

## build an array only containing the last 3 elements from the "a" array
b = a[-3:]
print(b)

## build an array only containing the 4th, 5th, and 6th elements of a
c = a[3:6]
print(c)

## buld an array skipping the first 12 elements
d = a[12:]
print(d)

## derrive an array from every other element in a
e = a[::2]
print(e)

## reverse a
reversed_a = np.flip(a)
print(reversed_a)

## build a 3, 3, 3 matrix with random floats
f = random((3, 3, 3))
print(f)

## build a vector of size 9 with values from 0 - 100 evenly spaced
g = np.linspace(0, 100, num=9, endpoint=True, retstep=False, dtype=None, axis=0)
print(g)

## build a vector of size 9 with values from -3 - 3
h = np.linspace(-3, 3, num=9, endpoint=True, retstep=False, dtype=None, axis=0)
print(h)

## graph the two vectors(g,h) on a line graph
plt.plot(g, h)
plt.show()

## Use NumPy to generate an array 128x128x3 with random values. Matplotlib .imshow() to display the array as an image.
i = random((128, 128, 3))
print(i.shape)
plt.imshow(i)

## vector addition and multiplication
v1 = np.array([4, 5, 2, 7])
v2 = np.array([2, 1, 3, 3])
add_v = v1 + v2
print(add_v) ## still a 1 dimentional array with 4 elements, but each element is the sum of the elements in the same index in the other two arrays
mult_v = v1 * v2
print(mult_v)

## vector broadcasting with scalar values
v3 = np.array([[1, 2, 3, 4], 
               [5, 6, 7, 8]])

add_v3 = v3 + 10 ## adds 10 to each element in the array
mult_v3 = v3 * 5 ## multiply each element by 5

a1 = np.array([[1, 3],
                [0, 1],
               [6, 2],
               [9, 7]])
     
b1 = np.array([[4, 1, 3],
               [5, 8, 5]])

matx_c = np.matmul(a1, b1)
print(f"Matrix c has {c.shape[0]} rows and {c.shape[1]} columns")
print(matx_c)

## import image of racoon and show the image
img = scipy.datasets.face()
plt.imshow(img)

## make the img black and white
sRGB_array = img / 255 ## the values must be divided by the rgb to bring them down to less than 1 so the math works out in the multiplication
grey_vals = np.array([0.2126, 0.7152, 0.0722]) ## the scalar values needed to properly find the correct grey values

grey_img = np.matmul(sRGB_array, grey_vals)
plt.imshow(grey_img, cmap='gray')

## rotate image by 180 degrees
reversed_grey = np.flip(grey_img)
plt.imshow(reversed_grey, cmap="gray")

## rotate colored img 90 degrees
img = scipy.datasets.face()
rotated_img = rotate(img, angle=90)
plt.imshow(rotated_img)

## invert the color scheme, solarize the img
inverted_color = 255 - img
plt.imshow(inverted_color)

## using a local image
file_name = 'yummy_macarons.jpg'
macaroon_img = Image.open(file_name)
img_array = np.array(macaroon_img)

sRGB_mac = img_array / 255 ## the values must be divided by the rgb to bring them down to less than 1 so the math works out in the multiplication
grey_mac = np.array([0.2126, 0.7152, 0.0722]) ## seems to adjust the contrast

grey_mac_img = np.matmul(sRGB_array, grey_vals)
plt.imshow(grey_mac_img, cmap='gray')

## modulate the colors  rotate around the color wheel
inverted_mac = 127 - img_array
plt.imshow(inverted_mac)