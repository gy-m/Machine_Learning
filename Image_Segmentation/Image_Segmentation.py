# import for OpenCV 3
import cv2
# import matplotlib
import matplotlib.pyplot as plt
# import numpy
# import numpy as np

# Color space conversions
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

# load the image 
nemo = cv2.imread('./Image_Segmentation/images/nemo0.jpg')

# plot the image in BGR color space (Default of opencV)
"""
plt.imshow(nemo)
plt.show()
"""

# plot the image in RGB color space
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
"""
plt.imshow(nemo)    # ?
plt.show()
"""

# convert an image from RGB to HSV
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

# Picking threshold 1
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

# Threshold Nemo - Mask for threshold 1
mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result = cv2.bitwise_and(nemo, nemo, mask=mask)

# Display nemo after mask 1
"""
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
"""

# Picking threshold 2
light_white = (0, 0, 200)
dark_white = (145, 60, 255)

mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

# Display nemo after mask 2
"""
plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()
"""

# Combining Masks
final_mask = mask + mask_white
final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)

# Display nemo after mask 1 + mask 2
"""
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()
"""

# Dealing with blur using GaussianBlur - Tradeoff: smoothing out image noise and reducing detail
blur = cv2.GaussianBlur(final_result, (7, 7), 0)

# Display nemo after mask 1 + mask 2 and after GaussianBlur
plt.imshow(blur)
plt.show()
