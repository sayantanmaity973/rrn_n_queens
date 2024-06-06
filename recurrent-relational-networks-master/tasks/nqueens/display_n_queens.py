import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#from rrn import NQueensRecurrentRelationalNet
from matplotlib import gridspec


def display(solution: np.array, fname: str):
    """
    Renders an N-Queens solution and saves it to a file

    :param solution: Numpy array representing the N-Queens solution
    :param fname: Where to save the rendered solution.
    """
    n = solution.shape[0]
    fig = plt.figure(figsize=(n, n))
    outer = gridspec.GridSpec(n, n, wspace=0.0, hspace=0.0)

    for i in range(n):
        for j in range(n):
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i, j], wspace=0.0, hspace=0.0)
            plt.subplot(inner[0])
            if solution[i] == j:
                plt.text(0.5, 0.5, 'Q', horizontalalignment='center', verticalalignment='center', fontsize=20)
            plt.xticks([])
            plt.yticks([])

            ax = fig.gca()
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_first_row():
                top = ax.spines['top']
                top.set_visible(True)
            if ax.is_last_row():
                bottom = ax.spines['bottom']
                bottom.set_visible(True)
            if ax.is_first_col():
                left = ax.spines['left']
                left.set_visible(True)
            if ax.is_last_col():
                right = ax.spines['right']
                right.set_visible(True)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    n = 8  # Change this to the size of the N-Queens board
    solution = np.random.randint(0, n, size=(n,))
    display(solution, 'n_queens_solution.pdf')
