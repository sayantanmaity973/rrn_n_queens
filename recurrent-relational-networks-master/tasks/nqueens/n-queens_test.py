import os

import matplotlib
import numpy as np

import sys
sys.path.insert(0, '../')

#from tasks.nqueens.display import board2logits, display
#from tasks.nqueens.rrn import NQueensRecurrentRelationalNet

from display_n_queens import board2logits, display
from RRN_four_queens import NQueensRecurrentRelationalNet

matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_dir = '//wsl.localhost/Ubuntu/home/rik/recurrent-relational-networks-master/tasks/4queens'
NQueensRecurrentRelationalNet.n_steps = n_steps = 64
render_steps = True

eval_fname = model_dir + '%d-eval.npz' % n_steps
if not os.path.exists(eval_fname):
    model = NQueensRecurrentRelationalNet(True)

    model.load(model_dir + "best")
    boards, logits, solutions, i = [], [], [], 0
    try:
        while True:
            board, logit, solution = model.test_batch()
            boards.append(board)
            logits.append(logit)
            solutions.append(solution)
            i += 1
            print(i)
    except Exception as e:  # It should throw a tf.errors.OutOfRangeError, but sometimes it throws another exception, so we'll catch em all
        np.savez(eval_fname, logits=logits, boards=boards, solutions=solutions)
        print(e)

data = np.load(eval_fname)
boards = data['boards'].reshape(-1, NQueensRecurrentRelationalNet.board_size)
logits = data['logits']  # n_batches, n_steps, batch_size, board_size, 3
solutions = data['solutions'].reshape(-1, NQueensRecurrentRelationalNet.board_size)

accuracy = np.mean(np.equal(np.argmax(logits[:, -1, ...], axis=-1), solutions))

print("Accuracy: %.2f%%" % (accuracy * 100))

print("Rendering steps... This will take a long time.")
if render_steps:
    idx = np.random.choice(len(boards), size=min(10, len(boards)), replace=False)
    boards = boards[idx]
    logits0 = board2logits(boards)
    logits = np.transpose(logits, (0, 2, 1, 3, 4))  # (n_batches, n_steps, batch_size, board_size, 3)
    logits = logits.reshape(-1, n_steps, NQueensRecurrentRelationalNet.board_size, 3)  # (n_batch*batch_size, n_steps, board_size, 3)
    logits = logits[idx]
    logits = np.concatenate([logits0[:, np.newaxis, ...], logits], axis=1)  # (N, n_steps+1, board_size, 3)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            display(logits[i, j], model_dir + "%03d-%02d.pdf" % (i, j))
