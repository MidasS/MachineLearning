import numpy as np
import matplotlib.pyplot as plt

s = np.sin(2* np.pi* 0.125* np.arange(20))
plt.plot(s, 'ro-')
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.show()