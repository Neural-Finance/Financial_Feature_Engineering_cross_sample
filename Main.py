from lenet import LENET
from datetime import datetime
import time
from Data_processing import *
from hyper_parameters import *
import pandas as pd


class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        pass


    def placeholders(self,shape1,shape2,shape3):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.keep_rate= tf.placeholder(tf.float32)

        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, shape1,shape2, shape3])
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size,shape1])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,shape1,shape2, shape3])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,shape1])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        model=LENET()
        self.logits = model.inference(self.image_placeholder, num_classes=self.image_placeholder.get_shape().as_list()[1])
        self.pred=tf.nn.softmax(self.logits)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.full_loss = self.loss(self.logits, self.label_placeholder)

        #predictions = logits
        #self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)
        #self.train_top1_error=self.full_loss
        self.train_op = self.train_operation(global_step, self.full_loss)


    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        #labels = tf.cast(labels, tf.float32)
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        cross_entropy=tf.losses.mean_squared_error(logits, labels)

        return cross_entropy

    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(80 - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        #batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        #batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size,:]

        return batch_data, batch_label


    def train_operation(self, global_step, total_loss):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.9, beta2=0.999, name='opt')
        train_op = opt.minimize(total_loss)

        return train_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)


    def train(self):
        '''
        This is the main function for training
        '''
        # For the first step, we are loading all training images and validation images into the
        # memory
        all_data, all_labels,vali_data, vali_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        shape1,shape2,shape3=all_data.shape[1],all_data.shape[2],all_data.shape[3]
        self.placeholders(shape1,shape2,shape3)

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print ('Restored from checkpoint...')
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print ('Start training...')
        print ('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                        FLAGS.train_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            start_time = time.time()

            _, train_loss_value,pred_ = sess.run([self.train_op,self.full_loss,self.logits],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr,
                                 self.keep_rate: 1.0})

            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                _, train_loss_value = sess.run([self.train_op,
                                                                   self.full_loss],
                                                                  {self.image_placeholder: train_batch_data,
                                                                   self.label_placeholder: train_batch_labels,
                                                                   self.lr_placeholder: FLAGS.init_lr,
                                                                   self.keep_rate: 1.0})

                print("Current step is %s, training loss is %s"%(str(step),str(train_loss_value)))

            if step==FLAGS.train_steps-1:
                _, train_loss_value,pred_ = sess.run([self.train_op,
                                                self.full_loss,self.logits],
                                               {self.image_placeholder: train_batch_data,
                                                self.label_placeholder: train_batch_labels,
                                                self.lr_placeholder: FLAGS.init_lr,
                                                self.keep_rate: 1.0})

                print("Current step is %s, training loss is %s" % (str(step), str(train_loss_value)))


# Initialize the Train object
train = Train()
# Start the training session
train.train()




