import json
import os
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
pair_path = os.path.join(abs_path,'cam_pair.json')

with open(pair_path,'r') as f:
    cam_pair = json.load(f)


data= np.array(cam_pair['45']['46']['time_pair']).reshape(-1, 1)

kde = KernelDensity(kernel='gaussian', bandwidth=5.0)
kde.fit(data)

# Creating an array of values for which we want to evaluate the density
x_d = np.linspace(min(data) - 20, max(data) + 20, 100).reshape(-1, 1)

# Computing the log density model (logarithm of the density)
log_density = kde.score_samples(x_d)
print(np.exp(log_density))
# print(np.exp(kde.score_samples(np.array(25).reshape(-1,1))))
# Plotting the density estimation
plt.figure(figsize=(8, 4))
plt.fill_between(x_d.flatten(), np.exp(log_density), alpha=0.5)
plt.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1)
plt.title('Kernel Density Estimation')
plt.xlabel('Data Values')
plt.ylabel('Density')
plt.show()