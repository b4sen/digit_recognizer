import tensorflow as tf
import pickle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
pickle.dump(x_train, open('../../data/x_train','wb'))
pickle.dump(y_train, open('../../data/y_train','wb'))
pickle.dump(x_test, open('../../data/x_test','wb'))
pickle.dump(y_test, open('../../data/y_test','wb'))
