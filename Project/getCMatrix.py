import tensorflow as tf
import numpy as np

class GetCMatrix(tf.keras.layers.Layer):
    def __init__(self,no_of_super_nodes,
                 collapse_regularization=0.1,
                 do_unpooling = False):
        super(GetCMatrix, self).__init__()
        self.no_of_superNodes=no_of_super_nodes
        self.collapse_regularization= collapse_regularization
        self.do_unpooling=do_unpooling

    def build(self, input_shape):
        self.n_digits = input_shape[0][-1]
        self.n_space=tf.constant([9 for i in range(self.n_digits)],dtype=tf.float32)
        self.weight= tf.constant(np.array([10**i for i in range(self.n_digits)],dtype=np.float32).reshape(-1,1))
        super().build(input_shape)

    def call(self, inputs):
        inputs,adjacency=inputs
        inputs=inputs*self.n_space
        supernode_index=tf.matmul(inputs,self.weight)
        assignments=supernode_index
        cluster_sizes = tf.math.reduce_sum(assignments, axis=0)  # Size [k].

        assignments_pooling = assignments / cluster_sizes  # Size [n, k].

        degrees = tf.sparse.reduce_sum(adjacency, axis=0)  # Size [n].
        degrees = tf.reshape(degrees, (-1, 1))

        number_of_nodes = adjacency.shape[1]
        number_of_edges = tf.math.reduce_sum(degrees)

        # Computes the size [k, k] pooled graph as S^T*A*S in two multiplications.
        graph_pooled = tf.transpose(
            tf.sparse.sparse_dense_matmul(adjacency, assignments))
        graph_pooled = tf.matmul(graph_pooled, assignments)

        # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
        # in three matrix multiplications by first processing the left part S^T*d
        # and then multyplying it by the right part d^T*S.
        # Left part is [k, 1] tensor.
        normalizer_left = tf.matmul(assignments, degrees, transpose_a=True)
        # Right part is [1, k] tensor.
        normalizer_right = tf.matmul(degrees, assignments, transpose_a=True)

        # Normalizer is rank-1 correction for degree distribution for degrees of the
        # nodes in the original graph, casted to the pooled graph.
        normalizer = tf.matmul(normalizer_left,
                            normalizer_right) / 2 / number_of_edges
        spectral_loss = -tf.linalg.trace(graph_pooled -
                                        normalizer) / 2 / number_of_edges
        self.add_loss(spectral_loss)

        collapse_loss = tf.norm(cluster_sizes) / number_of_nodes * tf.sqrt(
            float(self.no_of_superNodes)) - 1
        self.add_loss(self.collapse_regularization * collapse_loss)

        features_pooled = tf.matmul(assignments_pooling, assignments, transpose_a=True)
        features_pooled = tf.nn.selu(features_pooled)
        if self.do_unpooling:
            features_pooled = tf.matmul(assignments_pooling, features_pooled)   
        return assignments
        
        
        return tf.squeeze(supernode_index,axis=0)

        
    
    