

import tensorflow as tf
#from tasks.queens.rrn import NQueensRecurrentRelationalNet
import sys
sys.path.insert(0, '../')
from RRN_four_queens import NQueensRecurrentRelationalNet
import trainer
'''
# Create an instance of the QueensRecurrentRelationalNet model
nqueens_model = NQueensRecurrentRelationalNet()

# Start training the model
trainer.train(nqueens_model)
'''

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    # Create an instance of the model
    nqueens_model = NQueensRecurrentRelationalNet()

    '''
    # Example to get a batch from the iterator
    for mode in ["train", "valid", "test"]:
        nqueens_model.mode = mode

        if mode == "train":
            batch = next(iter(nqueens_model.train))
        elif mode == "valid":
            batch = next(iter(nqueens_model.valid))
        else:
            batch = next(iter(nqueens_model.test))

        print(f"{mode.capitalize()} batch inputs shape: {batch[0].shape}, labels shape: {batch[1].shape}")

        '''
    # with tf.compat.v1.Session() as sess:
    #     # Initialize the iterators
    #     #sess.run(nqueens_model.train_iterator.initializer)
    #     #sess.run(nqueens_model.valid_iterator.initializer)
    #     #sess.run(nqueens_model.test_iterator.initializer)

    #     sess.run(nqueens_model.train_iterator.initializer)
    #     sess.run(nqueens_model.valid_iterator.initializer)
    #     sess.run(nqueens_model.test_iterator.initializer)

    #     # Handles for switching between iterators
    #     #train_handle = sess.run(nqueens_model.train_iterator.string_handle())
    #     #valid_handle = sess.run(nqueens_model.valid_iterator.string_handle())
    #     #test_handle = sess.run(nqueens_model.test_iterator.string_handle())

    #     train_handle = sess.run(nqueens_model.train_iterator.string_handle())
    #     valid_handle = sess.run(nqueens_model.valid_iterator.string_handle())
    #     test_handle = sess.run(nqueens_model.test_iterator.string_handle())

    #     # Example to get a batch from the iterator
    #     sess.run(nqueens_model.next_element, feed_dict={nqueens_model.handle: train_handle})
    #     train_batch = sess.run(nqueens_model.next_element, feed_dict={nqueens_model.handle: train_handle})
    #     valid_batch = sess.run(nqueens_model.next_element, feed_dict={nqueens_model.handle: valid_handle})
    #     test_batch = sess.run(nqueens_model.next_element, feed_dict={nqueens_model.handle: test_handle})

    #     print(f"Train batch inputs shape: {train_batch[0].shape}, labels shape: {train_batch[1].shape}")
    #     print(f"Valid batch inputs shape: {valid_batch[0].shape}, labels shape: {valid_batch[1].shape}")
    #     print(f"Test batch inputs shape: {test_batch[0].shape}, labels shape: {test_batch[1].shape}")

    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     sess.run(nqueens_model.train_iterator.initializer)
    #     sess.run(nqueens_model.valid_iterator.initializer)
    #     sess.run(nqueens_model.test_iterator.initializer)
        
    #     train_batch = sess.run(nqueens_model.get_next_batch(nqueens_model.train_iterator, nqueens_model.edge_keep_prob))
    #     valid_batch = sess.run(nqueens_model.get_next_batch(nqueens_model.valid_iterator, 1.0))
    #     test_batch = sess.run(nqueens_model.get_next_batch(nqueens_model.test_iterator, 1.0))

    #     print(f"Train batch inputs shape: {train_batch[0].shape}, labels shape: {train_batch[1].shape}")
    #     print(f"Valid batch inputs shape: {valid_batch[0].shape}, labels shape: {valid_batch[1].shape}")
    #     print(f"Test batch inputs shape: {test_batch[0].shape}, labels shape: {test_batch[1].shape}")