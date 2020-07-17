import tensorflow as tf

class Discriminator(object):
    def __init__(self, n_node, node_emb_init, positive, config):
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="embedding_discriminator", shape=self.node_emb_init.shape, trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias  

        if positive == True:
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) +
                    tf.nn.l2_loss(self.node_embedding) +
                    tf.nn.l2_loss(self.bias))
        else:
            self.loss = tf.reduce_sum(
                - tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) +
                    tf.nn.l2_loss(self.node_embedding) +
                    tf.nn.l2_loss(self.bias))

        self.target_node = tf.placeholder(tf.int32, shape=[None])
        self.target_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.target_node)
        self.target_score = tf.matmul(self.target_embedding, self.embedding_matrix, transpose_b=True) + self.bias_vector 

        optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)

        self.d_updates = optimizer.minimize(self.loss)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))