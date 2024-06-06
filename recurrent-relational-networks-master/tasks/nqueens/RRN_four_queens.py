import os
import tensorflow as tf
from typing import Tuple
from tensorflow.keras import layers
import csv
import numpy as np

from tensorflow.keras import layers, regularizers
tf.config.run_functions_eagerly(True)
from tensorflow.keras import layers
from tensorflow.data import Dataset, Iterator
from tensorflow.nn import softmax
from tensorflow.python.ops.rnn_cell import LSTMCell

import sys
sys.path.insert(0, '../../')
import util
from message_passing import message_passing
from model import Model

class NQueensRecurrentRelationalNet(Model):
    devices = util.get_devices()
    batch_size = 16
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    emb_size = 16
    n_steps = 32
    edge_keep_prob = 1.0
    n_hidden = 96
    edges = 'n_queens'

    #def __init__(self)
    #def __init__(self, , train_csv: str, valid_csv: str, test_csv: str)
    def __init__(self, train_csv="train.csv", valid_csv="valid.csv", test_csv="test.csv"):
        super().__init__()
        #print ('Hola')
        #model = RecurrentRelationalNetwork(train_csv="//wsl.localhost/Ubuntu/home/rik/recurrent-relational-networks-master/tasks/4queens/train.csv", valid_csv="//wsl.localhost/Ubuntu/home/rik/recurrent-relational-networks-master/tasks/4queens/valid.csv", test_csv="//wsl.localhost/Ubuntu/home/rik/recurrent-relational-networks-master/tasks/4queens/test.csv")
        #train_csv = "home/rik/recurrent-relational-networks-master/tasks/nqueens/train.csv" 
        #valid_csv = "home/rik/recurrent-relational-networks-master/tasks/nqueens/valid.csv"
        #test_csv = "home/rik/recurrent-relational-networks-master/tasks/nqueens/test.csv"


        #self.is_testing = False
        #self.mode = "train"
        train_csv = "train.csv"
        test_csv = "test.csv"
        valid_csv = "valid.csv"

        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        self.is_testing = False
        #self.train, self.valid, self.test = self.encode.data()
        self.batch_size = 32  
        self.mode = "train"  # example mode

        print ( "Batch Size:" , self.batch_size )

        self.batch_size = 32  # Set your batch size here
        
        #self.train = self.load_csv(self.train_csv)
        #self.valid = self.load_csv(self.valid_csv)
        #self.test = self.load_csv(self.test_csv)
        

        self.train, self.valid, self.test = self.load_csv(self.train_csv, self.valid_csv, self.test_csv)
        print('Test train data on load:'. self.train)

        self.edge_keep_prob = 0.5  # Example probability value, adjust as needed

        self.devices = ['/gpu:0']  # Add your devices here
        self.emb_size = 128  # Example embedding size, adjust as needed

        #self.mode = "train"  # Initialize the mode
        '''
        self.train_iterator = iter(self.train)
        self.valid_iterator = iter(self.valid)
        self.test_iterator = iter(self.test)
        
        '''
        '''
        if self.is_testing:
            (quizzes, answers), edge_keep_prob = self.get_test_data()
        else:
            (quizzes, answers), edge_keep_prob = self.get_train_data()

        '''
        print('Test train data:', self.train[0])
        self.train_iterator = iter(self.train)
        self.valid_iterator = iter(self.valid)
        print("self.valid", self.valid)
        print("valid_iterator:", self.valid_iterator)
        #(quizzes, answers) = self.valid_iterator
        #print ("valid_iterator:", quizzes, answers)
        self.test_iterator = iter(self.test)
        
        print ("Hola 1, line before 79")
        print ("prev", tf.string)
        print ("tf is like:" , tf)

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_eager_execution()
        #self.mode = tf.compat.v1.placeholder(tf.string, shape=())
        self.mode = tf.compat.v1.placeholder(tf.string)
        
        print ("next", tf.string)
        print ("Hola2, line after 79")

        with tf.Graph().as_default(), tf.device('/cpu:0'):
        
            regularizer = tf.keras.regularizers.L2(1e-4)
            self.name = f"{self.revision} {self.message}"
            self.train, self.valid, self.test = self.load_csv(self.train_csv, self.valid_csv, self.test_csv)

            print("Building graph...")
            self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.optimizer = tf.optimizers.Adam(learning_rate=2e-4)

            self.mode = tf.compat.v1.placeholder(tf.string)

            # Create iterators
            #self.train_iterator = self.train.make_initializable_iterator()
            #self.valid_iterator = self.valid.make_initializable_iterator()
            #self.test_iterator = self.test.make_initializable_iterator()


            # Creating iterators
            self.train_iterator = tf.compat.v1.data.make_initializable_iterator(self.train)
            self.valid_iterator = tf.compat.v1.data.make_initializable_iterator(self.valid)
            self.test_iterator = tf.compat.v1.data.make_initializable_iterator(self.test)
            
            # Create iterators
            #self.train_iterator = tf.compat.v1.data.make_one_shot_iterator(self.train)
            #self.valid_iterator = tf.compat.v1.data.make_one_shot_iterator(self.valid)
            #self.test_iterator = tf.compat.v1.data.make_one_shot_iterator(self.test)

            self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
            #self.iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle, self.train.output_types, self.train.output_shapes)
            self.iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle, tf.compat.v1.data.get_output_types(self.train), tf.compat.v1.data.get_output_shapes(self.train))
            self.next_element = self.iterator.get_next()

            #codes
            if self.edges == 'n_queens':
                edges = self.n_queens_edges()
            elif self.edges == 'full':
                edges = [(i, j) for i in range(16) for j in range(16) if not i == j]
            else:
                raise ValueError('edges must be n_queens or full')

            edge_indices = tf.constant([(i + (b * 4), j + (b * 4)) for b in range(self.batch_size) for i, j in edges], tf.int32)
            n_edges = tf.shape(edge_indices)[0]
            edge_features = tf.zeros((n_edges, 1), tf.float32)
            positions = tf.constant([[(i, j) for i in range(4) for j in range(4)] for b in range(self.batch_size)], tf.int32)  # (bs, 16, 2)
            #rows = layers.Embedding(4, self.emb_size, input_length=16, name='row-embeddings', embeddings_regularizer=regularizer)(positions[:, :, 0])  # bs, 16, emb_size
            #cols = layers.Embedding(4, self.emb_size, input_length=16, name='cols-embeddings', embeddings_regularizer=regularizer)(positions[:, :, 1])  # bs, 16, emb_size

            rows = tf.random.normal([self.batch_size, 16, self.emb_size])  # Placeholder
            cols = tf.random.normal([self.batch_size, 16, self.emb_size])  # Placeholder            

            def avg_n(x):
                return tf.reduce_mean(tf.stack(x, axis=0), axis=0)

            towers = []
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
                for device_nr, device in enumerate(self.devices):
                    with tf.device('/cpu:0'):

                        '''
                        if self.is_testing:
                            (quizzes, answers), edge_keep_prob = self.test.get_next(), 1.0
                            edge_keep_prob = 1.0
                        else:
                            (quizzes, answers), edge_keep_prob = tf.cond(
                                tf.equal(self.mode, "train"),
                                true_fn=lambda: (self.train.get_next(), self.edge_keep_prob),
                                false_fn=lambda: (self.valid.get_next(), 1.0)
                            )
                        '''
                        
                        if self.is_testing:
                            quizzes, answers, edge_keep_prob = self.get_next_batch(self.test_iterator, 1.0)
                        else:
                            if self.mode == "train":
                                quizzes, answers, edge_keep_prob = self.get_next_batch(self.train_iterator, self.edge_keep_prob)
                            else:
                                quizzes, answers, edge_keep_prob = self.get_next_batch(self.valid_iterator, 1.0)

                        '''
                        if self.is_testing:
                            #quizzes, answers = next(iter(self.test))
                            (quizzes, answers) = self.test_iterator
                            edge_keep_prob = 1.0
                        else:
                            if self.mode == "train":
                                #quizzes, answers = next(iter(self.train))
                                (quizzes, answers) = self.train_iterator
                                edge_keep_prob = self.edge_keep_prob
                            else:
                                print ("Hello pre Valid")
                                (quizzes, answers) = next(iter(self.valid))
                                print ("Quizzes:", quizzes, "Answers:", answers)    
                                print('valid iteration:', self.valid_iterator)
                                (quizzes, answers) = self.valid_iterator
                                
                                edge_keep_prob = 1.0
                                print ("Hello post valid")
                                print ("Quizzes:", quizzes, "Answers:", answers)    
                        '''



                        x = layers.Embedding(5, self.emb_size, input_length=16, name='nr-embeddings', embeddings_regularizer=regularizer)(quizzes)  # bs, 16, emb_size
                        x = tf.concat([x, rows, cols], axis=2)
                        x = tf.reshape(x, (-1, 3 * self.emb_size))

                    rows = tf.random.normal([self.batch_size, 16, self.emb_size])  # Placeholder
                    cols = tf.random.normal([self.batch_size, 16, self.emb_size])  # Placeholder   

                    with tf.device(device), tf.compat.v1.name_scope("device-%s" % device_nr):

                        def mlp(x, scope):
                            with tf.compat.v1.variable_scope(scope):
                                for i in range(3):
                                    x = layers.Dense(self.n_hidden, activation='relu', kernel_regularizer=regularizer)(x)
                                return layers.Dense(self.n_hidden, activation=None, kernel_regularizer=regularizer)(x)

                        x = mlp(x, 'pre-fn')
                        x0 = x
                        n_nodes = tf.shape(x)[0]
                        outputs = []
                        log_losses = []
                        with tf.compat.v1.variable_scope('steps'):
                            lstm_cell = LSTMCell(self.n_hidden)
                            state = lstm_cell.zero_state(n_nodes, tf.float32)

                            for step in range(self.n_steps):
                                x = message_passing(x, edge_indices, edge_features, lambda x: mlp(x, 'message-fn'), edge_keep_prob)
                                x = mlp(tf.concat([x, x0], axis=1), 'post-fn')
                                x, state = lstm_cell(x, state)

                                with tf.compat.v1.variable_scope('graph-sum'):
                                    out = layers.Dense(5, activation=None)(x)
                                    out = tf.reshape(out, (-1, 16, 5))
                                    outputs.append(out)
                                    log_losses.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers, logits=out)))

                                tf.compat.v1.get_variable_scope().reuse_variables()

                        reg_loss = sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
                        loss = avg_n(log_losses) + reg_loss

                        towers.append({
                            'loss': loss,
                            'grads': self.optimizer.compute_gradients(loss),
                            'log_losses': tf.stack(log_losses),  # (n_steps, 1)
                            'quizzes': quizzes,  # (bs, 16, 5)
                            'answers': answers,  # (bs, 16, 5)
                            'outputs': tf.stack(outputs)  # n_steps, bs, 16, 5
                        })

                        tf.compat.v1.get_variable_scope().reuse_variables()

            self.loss = avg_n([t['loss'] for t in towers])
            self.out = tf.concat([t['outputs'] for t in towers], axis=1)  # n_steps, bs, 16, 5
            self.predicted = tf.cast(tf.argmax(self.out, axis=3), tf.int32)
            self.answers = tf.concat([t['answers'] for t in towers], axis=0)
            self.quizzes = tf.concat([t['quizzes'] for t in towers], axis=0)

            tf.compat.v1.summary.scalar('losses/total', self.loss)
            tf.compat.v1.summary.scalar('losses/reg', reg_loss)
            log_losses = avg_n([t['log_losses'] for t in towers])

            for step in range(self.n_steps):
                equal = tf.equal(self.answers, self.predicted[step])

                digit_acc = tf.reduce_mean(tf.cast(equal, tf.float32))
                tf.compat.v1.summary.scalar('steps/%d/digit-acc' % step, digit_acc)

                puzzle_acc = tf.reduce_mean(tf.cast(tf.reduce_all(equal, axis=1), tf.float32))
                tf.compat.v1.summary.scalar('steps/%d/puzzle-acc' % step, puzzle_acc)

                tf.compat.v1.summary.scalar('steps/%d/losses/log' % step, log_losses[step])

            avg_gradients = util.average_gradients([t['grads'] for t in towers])
            self.train_step = self.optimizer.apply_gradients(avg_gradients, global_step=self.global_step)

            self.session.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            util.print_vars(tf.compat.v1.trainable_variables())

            self.train_writer = tf.compat.v1.summary.FileWriter('/tmp/tensorboard/n_queens/%s/train/%s' % (self.revision, self.name), self.session.graph)
            self.test_writer = tf.compat.v1.summary.FileWriter('/tmp/tensorboard/n_queens/%s/test/%s' % (self.revision, self.name), self.session.graph)
            self.summaries = tf.compat.v1.summary.merge_all()

    def n_queens_edges(self) -> list[Tuple[int, int]]:
        def cross(a):
            return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

        idx = np.arange(16).reshape(4, 4)
        rows, columns, squares = [], [], []
        for i in range(4):
            rows += cross(idx[i, :])
            columns += cross(idx[:, i])
        for i in range(2):
            for j in range(2):
                squares += cross(idx[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2])
        return list(set(rows + columns + squares))

    @tf.function
    def get_next_batch(self, iterator, edge_keep_prob):
        quizzes, answers = next(iterator)
        return quizzes, answers, edge_keep_prob
    '''
    def load_csv(self, file_path):
        # Implement CSV loading logic here
        # For now, returning a dummy dataset
        data = tf.data.Dataset.from_tensor_slices((tf.random.uniform([100, 16], maxval=5, dtype=tf.int32), tf.random.uniform([100, 16], maxval=5, dtype=tf.int32)))
        data = data.batch(self.batch_size)
        return data
    '''
    def load_csv(self, train_csv: str, valid_csv: str, test_csv: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    #def load_csv(self, train_csv: str, valid_csv: str, test_csv: str) -> Tuple[Iterator, Iterator, Iterator]:
        '''
        def parse_csv(file_path):
            data = np.loadtxt(file_path, delimiter=",", dtype=np.int32, skiprows=1)
            return data[:, :-1], data[:, -1]
        '''
        
        def parse_csv(file_path):
            try:
                data = np.loadtxt(file_path, delimiter=",", dtype=np.int32, skiprows=1)
                features = data[:, :-1]
                labels = data[:, -1]
                return features, labels
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None, None

        #if train_data[0] is None or valid_data[0] is None or test_data[0] is None:
            #raise ValueError("Error in loading CSV files. Check the files and their formats.")


        train_data = parse_csv(train_csv)
        valid_data = parse_csv(valid_csv)
        test_data = parse_csv(test_csv)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(self.batch_size)
        #train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(self.batch_size * 10).repeat().batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).batch(self.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(self.batch_size)
        
        #train_dataset = Dataset.from_tensor_slices(train_data).shuffle(self.batch_size * 10).repeat(-1).batch(self.batch_size).make_one_shot_iterator()
        #valid_dataset = Dataset.from_tensor_slices(valid_data).shuffle(self.batch_size * 10).repeat(-1).batch(self.batch_size).make_one_shot_iterator()
        #test_dataset = Dataset.from_tensor_slices(test_data).shuffle(self.batch_size * 10).repeat(1).batch(self.batch_size).make_one_shot_iterator()

        #train_iterator = iter(train_dataset)
        #valid_iterator = iter(valid_dataset)
        #test_iterator = iter(test_dataset)

        return train_dataset, valid_dataset, test_dataset
    
# Example usage:
# model = YourModelClass(batch_size=32)
# train_dataset, valid_dataset, test_dataset = model.load_csv("train.csv", "valid.csv", "test.csv")
    
    ''' 
if __name__ == '__main__':
    # Create an instance of the model
    nqueens_model = NQueensRecurrentRelationalNet()

    # Example to get a batch from the iterator
    with tf.compat.v1.Session() as sess:
        train_batch = sess.run(nqueens_model.train_iterator.get_next())
        valid_batch = sess.run(nqueens_model.valid_iterator.get_next())
        test_batch = sess.run(nqueens_model.test_iterator.get_next())

        print(f"Train batch inputs shape: {train_batch[0].shape}, labels shape: {train_batch[1].shape}")
        print(f"Valid batch inputs shape: {valid_batch[0].shape}, labels shape: {valid_batch[1].shape}")
        print(f"Test batch inputs shape: {test_batch[0].shape}, labels shape: {test_batch[1].shape}")

    # Example to get a batch from the iterator
    for mode in ["train", "valid", "test"]:
        nqueens_model.mode = mode

        
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


    '''
    def load_csv(self, train_csv, valid_csv, test_csv) -> Tuple[Iterator, Iterator, Iterator]:
        def load_and_encode(csv_file):
            data = np.loadtxt(csv_file, delimiter=',')
            features = data[:, :-1]
            labels = data[:, -1]
            return Dataset.from_tensor_slices((features, labels))

        print("Loading and encoding data...")
        train = load_and_encode(train_csv).shuffle(self.batch_size * 10).repeat().batch(self.batch_size).make_one_shot_iterator()
        valid = load_and_encode(valid_csv).repeat().batch(self.batch_size).make_one_shot_iterator()
        test = load_and_encode(test_csv).batch(self.batch_size).make_one_shot_iterator()

        return train, valid, test

    '''



    def save(self, name: str):
        self.saver.save(self.session, name)

    def load(self, name: str):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def train_batch(self) -> float:
        _, _loss, _logits, _summaries, _step = self.session.run([self.train_step, self.loss, self.out, self.summaries, self.global_step], {self.mode: 'train'})
        if _step % 1000 == 0:
            self.train_writer.add_summary(_summaries, _step)

        return _loss

    def test_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _quizzes, _logits, _answers = self.session.run([self.quizzes, self.out, self.answers], {self.mode: 'foo'})
        return _quizzes, _logits, _answers

    def val_batch(self) -> float:
        _loss, _predicted, _answers, _summaries, _step = self.session.run([self.loss, self.predicted, self.answers, self.summaries, self.global_step], {self.mode: 'valid'})
        self.test_writer.add_summary(_summaries, _step)
        return _loss


