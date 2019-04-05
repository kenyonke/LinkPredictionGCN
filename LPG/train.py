from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import random
import time
from models import GCN
from utils import construct_feed_dict,preprocess_features,preprocess_adj,load_data,load_nell
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import os
import scipy.sparse as sp

'''
The code of implementation Graph Convolutional Network is developed by Thomas N. Kipf.
It can be found on https://github.com/tkipf/gcn.

@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
'''

# Train on CPU (hide GPU) due to memory constraints
# Of course, you can train the model with GPU which should be much faster.
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Set random seed
#seed = 1234
#np.random.seed(seed)
#tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'nell.0.001'
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('hidden1', 250, 'Number of units in hidden layer 1.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_string('operator', 'hadamard','Link features Operator') #'average', 'hadamard', 'weighted-l1', 'weighted-l2'

dataset = FLAGS.dataset
learning_rate = FLAGS.learning_rate
hidden1 = FLAGS.hidden1      # Number of units in hidden layer 0
dropout = FLAGS.dropout
weight_decay = 5e-4 # Weight for L2 loss
output_dim = 25

pos_test_path = dataset + '/pos_test.pkl'
neg_test_path = dataset + '/neg_test.pkl'
changedadj_path = dataset + '/changed_adj.pkl'
linkspath = dataset  + '/links.pkl'

# Load data
if dataset == 'nell.0.001':
    features = load_nell(dataset)[1]
else:
    features = load_data(dataset)[1]

with open(changedadj_path,'rb') as load_cha_adj:
    changed_adj = pickle.load(load_cha_adj) 

# Some preprocessing
if FLAGS.features == 0:
    changed_features = preprocess_features(changed_adj + sp.eye(changed_adj.shape[0]))
else:
    changed_features = preprocess_features(features)

support = [preprocess_adj(changed_adj)]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(changed_features[2], dtype=tf.int64)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32), 
    'length': tf.placeholder(tf.int64),
    'link_labels': tf.placeholder(shape=(None,2),dtype=tf.float32),
    'is_train' : tf.placeholder(tf.bool)
}


if dataset == 'cora': # 'cora', 'citeseer', 'pubmed'
    onehot_len = 2708
elif dataset == 'citeseer':
    onehot_len = 3327
elif dataset == 'pubmed':
    onehot_len = 19717 
elif dataset == 'nell.0.001':
    onehot_len = 65755

# load links data created before running embedding
with open(linkspath,'rb') as loadlinks:
    links = pickle.load(loadlinks)    

# shuffle trainning links
links_length = len(links)
lis = []
las = []
for link,label in links:
    lis.append(link)
    las.append(label)
c = list(zip(lis, las))
random.shuffle(c)
lis, las = zip(*c)
links = []
for i in range(len(las)):
    links.append((lis[i],las[i]))


# arguments of SparseTensor for training data
traina_indices = []
traina_values = []
trainb_indices = []
trainb_values = []
train_labels = np.zeros((links_length,2))

index = 0
for (row,col),label in links:
    
    traina_indices.append([index,row])
    traina_values.append(1)
    
    trainb_indices.append([index,col])
    trainb_values.append(1)
    
    if label == 1:
        train_labels[index][1] = 1
    else:
        train_labels[index][0] = 1
    index += 1
    
a_indices = np.array(traina_indices,dtype=np.int32)
a_values = np.array(traina_values,dtype=np.float32)
b_indices = np.array(trainb_indices,np.int32)
b_values = np.array(trainb_values,dtype=np.float32)


# load test links (list dtype) 
with open(pos_test_path,'rb') as load_pos_test:
    pos_test = pickle.load(load_pos_test)
with open(neg_test_path,'rb') as load_neg_test:
    neg_test = pickle.load(load_neg_test)
length_pos_test = len(pos_test)  
length_neg_test = len(neg_test)

# arguments of SparseTensor for test data
testa_indices = []
testa_values = []
testb_indices = []
testb_values = []
test_labels = np.zeros((length_pos_test + length_neg_test,2))

index = 0
for row,col in pos_test:
    testa_indices.append([index,row])
    testa_values.append(1)
    
    testb_indices.append([index,col])
    testb_values.append(1)
    
    test_labels[index][1] = 1
    index += 1

for row,col in neg_test:
    testa_indices.append([index,row])
    testa_values.append(1)
    
    testb_indices.append([index,col])
    testb_values.append(1)
    
    test_labels[index][0] = 1
    index += 1
testa_indices = np.array(testa_indices,dtype=np.int32)
testa_values = np.array(testa_values,dtype=np.float32)
testb_indices = np.array(testb_indices,dtype=np.int32)
testb_values = np.array(testb_values,dtype=np.float32)

#---------------------------------------------
#link prediction placeholders
a_indice = tf.placeholder(tf.int64)
a_value = tf.placeholder(tf.float32)
b_indice = tf.placeholder(tf.int64)
b_value = tf.placeholder(tf.float32)
length = tf.placeholder(tf.int64)

links_nodea = tf.SparseTensor(indices=a_indice, values=a_value, dense_shape=[placeholders['length'], onehot_len])
links_nodeb = tf.SparseTensor(indices=b_indice, values=b_value, dense_shape=[placeholders['length'], onehot_len])

# GCN model
model = model_func(placeholders, input_dim=changed_features[2][1],
                   output_dim=output_dim, learning_rate=learning_rate, logging=True)

# softmax classifier variables
sf_w = tf.Variable(tf.random_normal(shape=[output_dim,2]),dtype=tf.float32, name = 'W')
sf_b = tf.Variable(tf.zeros(shape=[1,2])+0.1,dtype=tf.float32, name = 'B')

# create link feautrues in a matrix
#(|links|,|V|) * (|V|,|E|) = (|links|,|V|) 
if FLAGS.operator == 'hadamard':
    links_features = tf.nn.relu(tf.multiply(tf.sparse_tensor_dense_matmul(links_nodea, model.outputs),
                             tf.sparse_tensor_dense_matmul(links_nodeb, model.outputs)))
elif FLAGS.operator == 'average':
    links_features = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(links_nodea, model.outputs),
                             tf.sparse_tensor_dense_matmul(links_nodeb, model.outputs))/2 )
elif FLAGS.operator == 'weighted-l1':
    links_features = tf.abs(tf.subtract(tf.sparse_tensor_dense_matmul(links_nodea, model.outputs),
                             tf.sparse_tensor_dense_matmul(links_nodeb, model.outputs)))
elif FLAGS.operator == 'weighted-l2':
    links_features = tf.square(tf.subtract(tf.sparse_tensor_dense_matmul(links_nodea, model.outputs),
                             tf.sparse_tensor_dense_matmul(links_nodeb, model.outputs)))

# Softmax based on link features
pred = tf.nn.softmax(tf.matmul(links_features, sf_w) + sf_b)
l2_norm = tf.reduce_sum(tf.square(sf_w))
weight_decay = tf.constant([weight_decay])

# sklearn AUC inputs
predictions_auc = pred[:,1]
labels_auc = tf.argmax(placeholders['link_labels'], axis=1)

# accuracy
acc = tf.metrics.accuracy(labels = tf.argmax(placeholders['link_labels'], axis=1),
                          predictions = tf.argmax(pred, axis=1),)[1]

cross_entropy = -tf.reduce_sum(placeholders['link_labels'] * tf.log(tf.clip_by_value(pred,1e-10,1.0)))
# loss & optimizer
loss = tf.reduce_mean(cross_entropy) + model.loss + weight_decay * l2_norm
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


s = time.time()
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Train model
    for epoch in range(FLAGS.epochs):
        #training
        # Construct feed dictionary
        feed_dict = construct_feed_dict(changed_features, support, links_length, train_labels, placeholders)
        feed_dict.update({placeholders['dropout'] : dropout,
                          placeholders['is_train'] : True})
        feed_dict.update({a_indice:a_indices,
                          a_value:a_values,
                          b_indice:b_indices,
                          b_value:b_values
                          })

        # Training step        
        _,loss_val,prediction = sess.run([optimizer,loss, pred],feed_dict=feed_dict)          
        print('epoch:', epoch, 'loss:', loss_val)
    print("Optimization Finished!")
    e = time.time()
    
    #test link prediction
    test_feed_dict = construct_feed_dict(changed_features, support, length_pos_test+length_neg_test, test_labels, placeholders)
    test_feed_dict.update({a_indice:testa_indices, a_value:testa_values,
                      b_indice:testb_indices, b_value:testb_values,
                      placeholders['is_train'] : False})
    
    prediction,accuracy,pred_,y_ = sess.run([pred,acc,predictions_auc,labels_auc],feed_dict=test_feed_dict)
    fpr, tpr, thresholds = metrics.roc_curve(y_, pred_, pos_label=1)
    precision, recall, thresholds = precision_recall_curve(y_, pred_)
    
    print('Sklearn-AUC:', metrics.auc(fpr, tpr))
    print('time:',(e-s)/FLAGS.epochs)
