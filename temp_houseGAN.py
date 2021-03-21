import numpy as np
import matplotlib.pyplot as plt
import math
from utils import visualize_house

house_npy = np.load('housegan_small.npy', allow_pickle=True)


idx_house = 1
house = house_npy[idx_house]
color_house = visualize_house(house)

plt.imshow(color_house)
plt.show()
