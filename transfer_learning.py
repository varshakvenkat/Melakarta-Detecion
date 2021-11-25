import tensorflow as tf
import pandas as pd
import numpy as np
import os
import librosa
import glob 
import sys

# Calculate MFCC values for the audio file passed as argument
def calculate_mfcc():
   file_name = sys.argv[5]
   try:
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None 
   return mfccs

# Normalize (min-max) the input dataframe
def normalize(dataset):
    values = list(dataset.columns.values)
    length=len(dataset)
    for each in values:
            maximum=max(dataset[each])
            minimum=min(dataset[each])
            column=dataset[each]
            newcol=[]
            for i in range(length):
                newcol.append((column[i]-minimum)/(maximum-minimum))
            dataset[each]=newcol


# Read base task data file
dataset = pd.read_csv(sys.argv[1])
normalize(dataset)
X=np.array(dataset)

# Read base task target file
dataset = pd.read_csv(sys.argv[2])
normalize(dataset)
y=np.array(dataset)

# Shuffle the dataset
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

# Split dataset into train(80%) and test(20%)
test_size = round(len(X_values)*0.8)
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

sess = tf.Session()
interval = 5000
epoch = 35000

# Creation of MLP with hidden layer connected to two different target layers (base and transfer learning)

X_data = tf.placeholder(shape=[None, 40], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y_target_trans=tf.placeholder(shape=[None, 3], dtype=tf.float32)
    
    
hidden_layer_nodes = 45

w1 = tf.Variable(tf.random_normal(shape=[40,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # First Bias

w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,10])) # Hidden layer -> Outputs
b2 = tf.Variable(tf.random_normal(shape=[10]))   # Second Bias

w2_trans=tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3])) # Hidden layer -> Outputs
b2_trans=tf.Variable(tf.random_normal(shape=[3]))   # Second Bias

hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))
final_output_trans = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2_trans), b2_trans))
loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))
loss_trans = tf.reduce_mean(-tf.reduce_sum(y_target_trans * tf.log(final_output_trans), axis=0))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
optimizer_trans = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss_trans)

correct_prediction=final_output_trans

init = tf.global_variables_initializer()
sess.run(init)
    
# Training model with base task
print('Training the model...')
for i in range(1, (epoch + 1)):
        sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
        if i % interval == 0:
            print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))    
            
# Testing model for base task
correctcount=0
counter=0
for i in range(len(X_test)):
        counter+=1
        output=np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]}))[0]
        if np.all(output==y_test[i]):
            correctcount+=1
print("Prediction Accuracy :",100*correctcount/len(y_test))

# Read transfer task data file
dataset = pd.read_csv(sys.argv[3])
normalize(dataset)
X=np.array(dataset)

# Read transfer task target file
dataset = pd.read_csv(sys.argv[4])
normalize(dataset)
y=np.array(dataset)

# Shuffle dataset
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

# Split dataset into train(80%) and test(20%)
test_size = round(len(X_values)*0.8)
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

    
# Training model with transfer task
print('Training the model...')
for i in range(1, (epoch + 1)):
        sess.run(optimizer_trans, feed_dict={X_data: X_train, y_target_trans: y_train})
        if i % interval == 0:
            print('Epoch', i, '|', 'Loss:', sess.run(loss_trans, feed_dict={X_data: X_train, y_target_trans: y_train}))    
            
# Testing model for transfer task
correctcount=0
counter=0
for i in range(len(X_test)):
        counter+=1
        output=np.rint(sess.run(final_output_trans, feed_dict={X_data: [X_test[i]]}))[0]
        if np.all(output==y_test[i]):
            correctcount+=1
print("Prediction Accuracy :",100*correctcount/len(y_test))    

# Prediction of audio file passed as argument
output=correct_prediction.eval( session = sess,
        feed_dict={
            X_data: [calculate_mfcc()]}
    )
print(output)
sess.close()