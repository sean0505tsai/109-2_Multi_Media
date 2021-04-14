import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

image = data.astronaut()

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16), 
                    cells_per_block=(16, 16), visualize=True, multichannel=True)
'''
orientations:    Number of orientation bins, default = 9
pixels_per_cell: set the size of Cell, default = (8, 8)
cells_per_block: set the size of Block(正規化範圍), default = (3, 3)
block_norm:      the method of normalization, L1, L1-sqrt, L2, L2-Hys
visualize:       decide whether to return HOG feature image, default = False
multichannel:    將影像視為3通道(True) 單通道(false), default = None
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap = plt.cm.gray)
ax1.set_title('Imput image')

#Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap = plt.cm.gray)
ax2.set_title('Histogram of oriented Gradients')
plt.show()