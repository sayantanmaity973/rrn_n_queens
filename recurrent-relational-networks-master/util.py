import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def print_vars(vars):
    total = 0
    for var in vars:
        print(var.name, var.get_shape())
        total += np.prod(var.get_shape().as_list())
    print(total)


def get_devices():
    gpus = [x.name for x in (device_lib.list_local_devices()) if x.device_type == 'GPU']
    if len(gpus) > 0:
        devices = gpus
    else:
        print("WARNING: No GPU's found. Using CPU")
        devices = ['cpu:0']

    print("Using devices: ", devices)
    return devices


def average_gradients(tower_grads, name='avg-grads'):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    with tf.name_scope(name):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
