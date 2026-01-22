import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LogNorm, CenteredNorm

def derivative_gaussian_kernel_3d(sigma: float, direction='x'):
    t_size = int(np.ceil(3*sigma))
    xx = np.arange(-t_size, t_size+1, dtype=np.double)
    x, y, z = np.meshgrid(xx, xx, xx, indexing='xy')
    G = np.exp(-(x**2 + y**2 + z**2)/(2*sigma*sigma))
    G = G / np.sum(G)

    dGdx = -x * G
    dGdx = dGdx / np.sum(dGdx * -x)

    if direction == 'x':
        kernel = dGdx
    elif direction == 'y':
        kernel = np.transpose(dGdx, (2, 0, 1))
    elif direction == 'z':
        kernel = np.transpose(dGdx, (1, 2, 0))
    else:
        kernel = G
    return kernel, t_size

# Tests

def plot4d(data):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data != 0
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    ax.scatter(x, y, z, c=data.flatten(), s=20.0 * mask, norm=Normalize(), edgecolor="face", alpha=0.6, marker="s", cmap="magma", linewidth=0)
    plt.tight_layout()

# for dir in ['x', 'y', 'z', 'g']:
#     noyau, _ = derivative_gaussian_kernel_3d(1.2, direction=dir)
#     print(noyau.min(), noyau.max())
#     plot4d(noyau)
#     plt.show()

