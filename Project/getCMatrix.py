import tensorflow as tf
import numpy as np

class GetCMatrix(tf.keras.layers.Layer):
    def __init__(self,no_of_super_nodes):
        super(GetCMatrix, self).__init__()
        self.no_of_superNodes=no_of_super_nodes

    def build(self, input_shape):
        self.n_digits = input_shape[0][-1]
        self.n_space=tf.constant([9 for i in range(self.n_digits)],dtype=tf.float32)
        self.weight= tf.constant(np.array([10**i for i in range(self.n_digits)],dtype=np.float32).reshape(-1,1))
        super().build(input_shape)

    def call(self, inputs):
        
        inputs=inputs*self.n_space
        supernode_index=tf.matmul(inputs,self.weight)
        
        
        return tf.squeeze(supernode_index,axis=0)

        
    
    