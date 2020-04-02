# Sirohi, Krishnakant Singh
# 1001-668-969
# 2019-12_01
# Assignment-04-03

import tensorflow as tf
from cnn import CNN

batch_size = 64
num_classes = 10
epochs = 1
num_predictions = 20

(x_train, y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
model = CNN()
model.add_input_layer(x_train.shape[1:])
model.append_conv2d_layer(num_of_filters=32, kernel_size=3, padding='same', activation='relu')
model.append_conv2d_layer(num_of_filters=32, kernel_size=3, padding='same', activation='relu')
model.append_maxpooling2d_layer(pool_size=2)
model.append_conv2d_layer(num_of_filters=64, kernel_size=3, padding='same', activation='relu')
model.append_conv2d_layer(num_of_filters=64, kernel_size=3, padding='same', activation='relu')
model.append_maxpooling2d_layer(pool_size=2)
model.append_flatten_layer()
model.append_dense_layer(num_nodes=512, activation='relu')
model.append_dense_layer(num_nodes=num_classes, activation='softmax')
model.set_optimizer(optimizer='RMSprop', learning_rate=0.0001)
model.set_loss_function('categorical_crossentropy')
model.set_metric(['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.train(x_train, y_train, num_classes, epochs)
model.evaluate(x_test,y_test)