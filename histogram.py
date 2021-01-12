from skimage import io
import matplotlib.pyplot as plt
import numpy as np
image = io.imread('./Multi-label-classification/2696299_20201124_CR_1_1.jpg')
#  print(np.sort(image.ravel()))
ax = plt.hist(image.ravel(), bins = 256, density=True)
plt.show()
image = io.imread('./Multi-label-classification/6510414_20201209_CR_1_1.jpg')
#  print(np.sort(image.ravel()))
ax = plt.hist(image.ravel(), bins = 256, density=True)
plt.show()
image = io.imread('./Multi-label-classification/4249331_20201209_PX_1_1.jpg')
#  print(np.sort(image.ravel()))
ax = plt.hist(image.ravel(), bins = 256, density=True)
plt.show()

uniqueValues, occurCount = np.unique(image.ravel(), return_counts=True)
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)
print(image.ravel().shape)
total = image.ravel().shape[0]
#  class Histogram:
intensity = np.zeros(256)
print(intensity.shape)

for idx, cnt in zip(uniqueValues, occurCount):
    intensity[idx] = cnt / total
#  print(intensity)
#  plt.plot(range(256),intensity)
#  plt.show()
