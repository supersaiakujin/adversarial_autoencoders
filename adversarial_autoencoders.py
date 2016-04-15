# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os

n_epoch_pretraing = 10
n_epoch = 100
n_train = 50000
n_batch = 100
n_label = 10
imagesize = 28
latent_dim = 2
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

enable_pretraining = True
enable_load_pretraining_model = False
filename_pretraining_model = 'pretrain_model.ckpt'
filename_training_model = 'trainmodel.ckpt'
prefix_intermediated_training_model = 'trainmodel'
enable_softmax_cross_entropy = False

prior_sample_type = 'gaussian'
#prior_sample_type = 'swissroll'

# folder
folder_data = './data/mnist'
folder_log = './tmp'
folder_img = './img'
folder_decode = './dec'
folder_model = './model'


def get_10_2d_gaussian(labels, n_label, x_var=1.0, y_var=1.0, xoff=2.0):
    '''
    n_label = 10
    N = 10000
    labels = np.random.randint(0,n_label, N)

    z = get_10_2d_gaussian(labels, n_label, x_var=2.0, y_var=0.15, xoff=2.)

    import matplotlib.pyplot as plt

    plt.scatter(z[:,0], z[:,1])

    '''
    n_batch = len(labels)
    x = np.random.normal(0.0, x_var, (n_batch, 1))
    y = np.random.normal(0.0, y_var, (n_batch, 1))
    z = np.zeros((n_batch, 2), dtype=np.float32)
    
    c = np.zeros(n_label)
    s = np.zeros(n_label)
    
    for idx in range(n_label):
        c[idx] = np.cos(2.0 * np.pi * idx/n_label)
        s[idx] = np.sin(2.0 * np.pi * idx/n_label)

    one_hot_label = np.zeros((n_batch, n_label))
    for idx, (x_, y_, lbl) in enumerate(zip(x,y, labels)):
        z[idx][0] = (x_ + xoff) * c[lbl] - y_ * s[lbl]
        z[idx][1] = (x_ + xoff) * s[lbl] + y_ * c[lbl]
        one_hot_label[idx,lbl] = 1.0
        
    return z, one_hot_label

def get_swiss_roll(labels, n_label, r_var=3.0, n_roll=2.0, noise=0.0):
    '''
    z, _ = get_swiss_roll(lb, 10, r_var=10.0, n_roll=2.2, noise=0.3)
    '''

    n_batch = len(labels)
    u = np.random.uniform(0.0, 1.0, size=n_batch)
    uni = u / float(n_label) + labels.astype(np.float) / float(n_label)
    rad = np.pi * 2.0 * n_roll * np.sqrt(uni)
    r = np.sqrt(uni) * r_var 
    x = r * np.cos(rad) + noise * np.random.normal(0.0, 1.0, n_batch)
    y = r * np.sin(rad) + noise * np.random.normal(0.0, 1.0, n_batch)
    z = np.zeros((n_batch, 2), dtype=np.float32)
    z[:,0] = x
    z[:,1] = y
    
    one_hot_label = np.zeros((n_batch, n_label))
    for idx, lbl in enumerate(labels):
        one_hot_label[idx, lbl] = 1.0
        
    return z, one_hot_label

def get_prior_samples(labels, n_label, sample_type='gaussian'):
    if sample_type == 'gaussian':
        return get_10_2d_gaussian(labels, n_label, x_var=2.0, y_var=0.5, xoff=10.0)
    elif sample_type == 'swissroll':
        return get_swiss_roll(labels, n_label, r_var=15.0, n_roll=2.2, noise=0.5)
    
def plot_q_z(data, filename, n_label=10, xlim=None, ylim=None):
    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8,8))
    color = cm.rainbow(np.linspace(0,1,n_label))
    for l, c in zip(range(10), color):
        ix = np.where(data[:,2]==l)
        ax.scatter(data[ix,0], data[ix, 1], c=c, label=l, s=8, linewidth=0)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    plt.savefig(filename)
    plt.close()
    
def save_decode_images(img1, img2, filepath):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.savefig(filepath)
    plt.close()

def save_decode_image_array(x, filepath, cols = 10):
    rows = int(np.ceil(x.shape[0] / cols))
    
    plt.clf()
    fig, axes = plt.subplots(ncols=cols, nrows=rows)
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(x[r*cols+c,:].reshape(imagesize, imagesize))
            axes[r, c].set(adjustable='box-forced',aspect='equal')
            axes[r, c].get_xaxis().set_visible(False)
            axes[r, c].get_yaxis().set_visible(False)
    
    plt.savefig(filepath)
    plt.close()
    
def encoder(input, output_dim, training, stddev=0.02, bias_value=0, reuse=False):
    
    w1 = tf.get_variable("w1", [input.get_shape()[1],1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b1 = tf.get_variable("b1", [1000], initializer=tf.constant_initializer(bias_value))

    w2 = tf.get_variable("w2", [1000,1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b2 = tf.get_variable("b2", [1000], initializer=tf.constant_initializer(bias_value))

    w3 = tf.get_variable("w3", [1000,output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
    b3 = tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(bias_value))

    fc1 = tf.nn.relu(tf.matmul( input, w1 ) + b1, name='relu1')
    fc2 = tf.nn.relu(tf.matmul( fc1  , w2 ) + b2, name='relu2')
    fc3 = tf.matmul( fc2  , w3 ) + b3

    if not reuse:
        tf.histogram_summary('EN/L1/activation', fc1)
        tf.histogram_summary('EN/L1/weight'    , w1)
        tf.histogram_summary('EN/L1/bias'      , b1)
        tf.scalar_summary(   'EN/L1/sparsity'  , tf.nn.zero_fraction(fc1))
        
        tf.histogram_summary('EN/L2/activation', fc2)
        tf.histogram_summary('EN/L2/weight'    , w2)
        tf.histogram_summary('EN/L2/bias'      , b2)
        tf.scalar_summary(   'EN/L2/sparsity'  , tf.nn.zero_fraction(fc2))

        tf.histogram_summary('EN/L3/activation', fc3)
        tf.histogram_summary('EN/L3/weight'    , w3)
        tf.histogram_summary('EN/L3/bias'      , b3)
        tf.scalar_summary(   'EN/L3/sparsity'  , tf.nn.zero_fraction(fc3))
        
    return fc3, [w1, b1, w2, b2, w3, b3]

def decoder(input, output_dim, training, stddev=0.02, bias_value=0, reuse=False):
        
    w1 = tf.get_variable("w1", [input.get_shape()[1],1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b1 = tf.get_variable("b1", [1000], initializer=tf.constant_initializer(bias_value))

    w2 = tf.get_variable("w2", [1000,1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b2 = tf.get_variable("b2", [1000], initializer=tf.constant_initializer(bias_value))

    w3 = tf.get_variable("w3", [1000,output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
    b3 = tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(bias_value))

    fc1 = tf.nn.relu(tf.matmul( input, w1 ) + b1, name='relu1')
    fc2 = tf.nn.relu(tf.matmul( fc1  , w2 ) + b2, name='relu2')
    fc3 = tf.nn.sigmoid(tf.matmul( fc2  , w3 ) + b3 )

    if not reuse:
        tf.histogram_summary('DE/L1/activation', fc1)
        tf.histogram_summary('DE/L1/weight'    , w1)
        tf.histogram_summary('DE/L1/bias'      , b1)
        tf.scalar_summary(   'DE/L1/sparsity'  , tf.nn.zero_fraction(fc1))
        
        tf.histogram_summary('DE/L2/activation', fc2)
        tf.histogram_summary('DE/L2/weight'    , w2)
        tf.histogram_summary('DE/L2/bias'      , b2)
        tf.scalar_summary(   'DE/L2/sparsity'  , tf.nn.zero_fraction(fc2))

        tf.histogram_summary('DE/L3/activation', fc3)
        tf.histogram_summary('DE/L3/weight'    , w3)
        tf.histogram_summary('DE/L3/bias'      , b3)
        tf.scalar_summary(   'DE/L3/sparsity'  , tf.nn.zero_fraction(fc3))
        
    return fc3, [w1, b1, w2, b2, w3, b3]

def discriminator(z, lbl, output_dim, training, stddev=0.02, bias_value=0, reuse=False):

    new_z = tf.concat(1, [z, lbl], name='dis_concat')
    
    w1 = tf.get_variable("w1", [new_z.get_shape()[1],1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b1 = tf.get_variable("b1", [1000], initializer=tf.constant_initializer(bias_value))

    w2 = tf.get_variable("w2", [1000,1000], initializer=tf.random_normal_initializer(stddev=stddev))
    b2 = tf.get_variable("b2", [1000], initializer=tf.constant_initializer(bias_value))

    w3 = tf.get_variable("w3", [1000,output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
    b3 = tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(bias_value))

    if enable_softmax_cross_entropy:
        fc1 = tf.nn.relu(tf.matmul( new_z, w1 ) + b1, name='relu1')
        fc2 = tf.nn.relu(tf.matmul( fc1  , w2 ) + b2, name='relu2')
        fc3 = tf.nn.relu(tf.matmul( fc2  , w3 ) + b3 )
    else:
        fc1 = tf.nn.relu(tf.matmul( new_z, w1 ) + b1, name='relu1')
        fc2 = tf.nn.relu(tf.matmul( fc1  , w2 ) + b2, name='relu2')
        fc3 = tf.nn.sigmoid(tf.matmul( fc2  , w3 ) + b3 )
    
    if not reuse:
        tf.histogram_summary('D/L1/activation', fc1)
        tf.histogram_summary('D/L1/weight'    , w1)
        tf.histogram_summary('D/L1/bias'      , b1)
        tf.scalar_summary(   'D/L1/sparsity'  , tf.nn.zero_fraction(fc1))
        
        tf.histogram_summary('D/L2/activation', fc2)
        tf.histogram_summary('D/L2/weight'    , w2)
        tf.histogram_summary('D/L2/bias'      , b2)
        tf.scalar_summary(   'D/L2/sparsity'  , tf.nn.zero_fraction(fc2))

        tf.histogram_summary('D/L3/activation', fc3)
        tf.histogram_summary('D/L3/weight'    , w3)
        tf.histogram_summary('D/L3/bias'      , b3)
        tf.scalar_summary(   'D/L3/sparsity'  , tf.nn.zero_fraction(fc3))
        
    return fc3, [w1, b1, w2, b2, w3, b3]
# check folder
if not os.path.exists(folder_log):
    print 'create %s' % folder_log
    os.mkdir(folder_log)
    
if not os.path.exists(folder_img):
    print 'create %s' % folder_img
    os.mkdir(folder_img)
    
if not os.path.exists(folder_decode):
    print 'create %s' % folder_decode
    os.mkdir(folder_decode)
    
if not os.path.exists(folder_model):
    print 'create %s' % folder_decode
    os.mkdir(folder_model)
    
# save prior samples
index_labels = np.random.randint(0,n_label, 10000)
samples, _ = get_prior_samples(index_labels, n_label, sample_type=prior_sample_type)

result = np.zeros((10000, 3))
result[:,0:2] = samples
result[:,2] = index_labels
plot_q_z(result, os.path.join(folder_img,'samples.png'), xlim=(-20,20),ylim=(-20,20))

with tf.Graph().as_default() as g:
    
    training_node = tf.placeholder(tf.bool, name='train')

    with tf.variable_scope("Encoder"):
        input_x = tf.placeholder(tf.float32, shape=(n_batch, imagesize*imagesize), name='x')
        learning_rate_enc = tf.placeholder(tf.float32, shape=[], name='learning_rate_enc')
        learning_rate_gen = tf.placeholder(tf.float32, shape=[], name='learning_rate_gen')
        q_z, theta_enc = encoder(input_x, latent_dim, training_node)

    with tf.variable_scope("Decoder") as scope:
        x_, theta_dec = decoder(q_z, imagesize*imagesize, training_node)
        
        input_sample_z = tf.placeholder(tf.float32, shape=(n_batch, latent_dim), name='sample_z')
        scope.reuse_variables()
        sample_x, theta_dec = decoder(input_sample_z,imagesize*imagesize, training_node, reuse=True)
                                    
    with tf.variable_scope("D") as scope:
        input_z = tf.placeholder(tf.float32, shape=(n_batch, latent_dim), name='z')
        input_l = tf.placeholder(tf.float32, shape=(n_batch, n_label), name='label')
        learning_rate_dis = tf.placeholder(tf.float32, shape=[], name='learning_rate_dis')
        if enable_softmax_cross_entropy:
            fc, theta_d = discriminator(input_z, input_l, 2, training_node, reuse=False)
            D1 = tf.maximum(fc, 0.01)
        else:
            fc, theta_d = discriminator(input_z, input_l, 1, training_node, reuse=False)
            D1 = tf.maximum(tf.minimum(fc, 0.99), 0.01)

        scope.reuse_variables()
        if enable_softmax_cross_entropy:
            fc, theta_d = discriminator(q_z, input_l, 2, training_node, reuse=True)
            D2 = tf.maximum(fc, 0.01)
        else:
            fc, theta_d = discriminator(q_z, input_l, 1, training_node, reuse=True)
            D2 = tf.maximum(tf.minimum(fc, 0.99), 0.01)

    ones = tf.Variable(tf.ones([n_batch], tf.int64), trainable=False)
    zeros = tf.Variable(tf.zeros([n_batch], tf.int64), trainable=False)
    
    loss_rec = tf.reduce_mean(tf.square(input_x - x_))
    if enable_softmax_cross_entropy:
        loss_adv = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(D1, ones, name='adv_log1')) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(D2, zeros, name='adv_log2'))
        loss_gen = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(D2, ones, name='gen_log1'))
    else:
        loss_adv = -tf.reduce_mean(tf.log(D1, name='adv_log1') + tf.log(1-D2, name='adv_log2'))
        loss_gen = -tf.reduce_mean(tf.log(D2, name='gen_log1'))
    
    tf.scalar_summary('loss_rec', loss_rec)
    tf.scalar_summary('loss_adv', loss_adv)
    tf.scalar_summary('loss_gen', loss_gen)

    varlists_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
    varlists_dec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder")
    varlists_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")
    
    print 'varlists_enc'
    for v in varlists_enc:
        print v.name
    print 'varlists_dec'
    for v in varlists_dec:
        print v.name
    print 'varlists_dis'
    for v in varlists_dis:
        print v.name
    print 'theta_enc'
    for v in theta_enc:
        print v.name
    print 'theta_dec'
    for v in theta_dec:
        print v.name
    print 'theta_d'
    for v in theta_d:
        print v.name

    theta_rec = theta_enc + theta_dec
    print 'theta_rec'
    for v in theta_rec:
        print v.name
        
    opt_rec = tf.train.AdamOptimizer(learning_rate=learning_rate_enc, beta1= beta1, beta2 = beta2, name='Adam_rec').minimize(loss_rec, var_list=theta_rec)
    opt_adv = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1= beta1, beta2 = beta2, name='Adam_adv').minimize(loss_adv, var_list=theta_d)
    opt_gen = tf.train.AdamOptimizer(learning_rate=learning_rate_gen, beta1= beta1, beta2 = beta2, name='Adam_gen').minimize(loss_gen, var_list=theta_enc)

    mnist = input_data.read_data_sets(folder_data, one_hot=False)

    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    writer = tf.train.SummaryWriter(folder_log, graph_def=sess.graph_def)

    hist_rec = []
    hist_adv = []
    step = 0

    _lr_enc = 0.001
    _lr_gen = 0.001
    _lr_dis = 0.0002

    if enable_pretraining:
        if enable_load_pretraining_model:
            saver.restore(sess, os.path.join(folder_model, filename_pretraining_model))
            print 'load pre training Model'
        else:
            for epoch in xrange(n_epoch_pretraing):
                for idx in xrange(0, n_train, n_batch):
                    x, index_labels = mnist.train.next_batch(n_batch)
                    feed_dict = {input_x: x, learning_rate_enc: _lr_enc, training_node: True}
                    _x, lr, _ = sess.run([x_, loss_rec, opt_rec], feed_dict = feed_dict)
            
                    if idx % 10000 == 0:
                        print 'epoch:%d, traing:%d, loss rec:%.8f' % (epoch, idx, lr)
                
                    step += 1
                if epoch == 50:
                    _lr_enc = 0.0002

            save_path = saver.save(sess, os.path.join(folder_model, filename_pretraining_model))
            print 'save  pre training model in file: %s' % save_path
    
    _lr_enc = 0.001
    _lr_gen = 0.001
    _lr_dis = 0.0002
    step = 0
    last_lg = 0
    for epoch in xrange(n_epoch):
        training_flag_rec = True
        training_flag_dis = True
        training_flag_gen = True
            
        for idx in xrange(0, n_train, n_batch):
            if step % 2 == 0:
                training_flag_rec = True
                training_flag_dis = True
                training_flag_gen = True
            else:
                training_flag_rec = False
                training_flag_dis = False
                training_flag_gen = True
                
            x, index_labels = mnist.train.next_batch(n_batch)
            p_z, one_hot_label = get_prior_samples(index_labels, n_label, sample_type=prior_sample_type)

            feed_dict = {input_x: x, input_z: p_z, input_l: one_hot_label, learning_rate_enc: _lr_enc, learning_rate_gen: _lr_gen, learning_rate_dis: _lr_dis, training_node: True}

            # reconstruction
            if training_flag_rec:
                lr, _ = sess.run([loss_rec, opt_rec], feed_dict = feed_dict)
        
            # discriminator
            if training_flag_dis:
                la, _ = sess.run([loss_adv, opt_adv], feed_dict = feed_dict)
    
            # generator
            if training_flag_gen:
                lg, _ = sess.run([loss_gen, opt_gen], feed_dict = feed_dict)

            last_lg = lg
            if step % 500 == 0:

                _x, summary_str = sess.run([x_, summary_op], feed_dict = feed_dict)
                writer.add_summary(summary_str, step)
                print 'epoch:%d, train: %d, rec: %.8f, adv: %.8f, gen: %.8f' % (epoch, idx, lr, la, lg)

                hist_rec.append(lr)
                hist_adv.append(la)

            step += 1

        for l in xrange(n_label):
            index_labels = np.ones(n_batch) * l
            p_z, one_hot_label = get_prior_samples(index_labels, n_label, sample_type=prior_sample_type)

            _x = sess.run(sample_x, feed_dict = {input_sample_z: p_z, training_node: False})
            save_decode_image_array(_x, os.path.join(folder_decode,'genimg_e%05d_d%02d.png' % (epoch, l)))

        if epoch == 50:
            _lr_enc = 0.001
        if epoch % 10 == 0:
            save_path = saver.save(sess, os.path.join(folder_model, '%s_epoch%04d.ckpt' % (prefix_intermediated_training_model, epoch)))
            print 'Model save in file: %s' % save_path
            
        result = np.zeros((10000, 3))
    
        for idx in xrange(0, 10000, n_batch):
            x, index_labels = mnist.test.next_batch(n_batch)
            _q_z = sess.run(q_z, feed_dict={input_x: x, training_node: False})
            result[idx:idx+n_batch,0:2] = _q_z
            result[idx:idx+n_batch,2] = index_labels

        plot_q_z(result, os.path.join(folder_img,'test_epoch%03d.png' % epoch), xlim=(-20,20),ylim=(-20,20))

    save_path = saver.save(sess, os.path.join(folder_model, filename_training_model))
    print 'Model save in file: %s' % save_path
        
