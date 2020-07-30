import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class MF:

    def __init__(self, n_user, n_item, lr, latent_dim, l2_weight):
        self._parse_args(n_user, n_item, lr, latent_dim, l2_weight)
        self._build_inputs()
        self._build_parameters()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, n_user, n_item, lr, latent_dim, l2_weight):
        self.n_user = n_user
        self.n_item = n_item
        self.lr = lr
        self.latent_dim = latent_dim
        self.l2_weight = l2_weight

    def _build_inputs(self):
        with tf.name_scope('inputs'):
            self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
            self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
            self.ratings = tf.placeholder(tf.float32, shape=[None], name='ratings')

    def _build_parameters(self):
        with tf.name_scope('parameters'):
            self.user_embeddings = tf.get_variable(
                'user_embeddings',
                shape=[self.n_user, self.latent_dim],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
            self.item_embeddings = tf.get_variable(
                'item_embeddings',
                shape=[self.n_item, self.latent_dim],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
            self.user_bias = tf.get_variable(
                'user_bias',
                shape=[self.n_user],
                dtype=tf.float32,
				initializer=tf.zeros_initializer())
            self.item_bias = tf.get_variable(
                'item_bias',
                shape=[self.n_item],
                dtype=tf.float32,
				initializer=tf.zeros_initializer())
            self.global_bias = tf.get_variable(
                'global_bias',
                shape=[],
                dtype=tf.float32,
				initializer=tf.zeros_initializer())

    def _build_model(self):
        with tf.name_scope('model'):
            batch_user_bias = tf.nn.embedding_lookup(self.user_bias, self.user_ids)
            batch_item_bias = tf.nn.embedding_lookup(self.item_bias, self.item_ids)

            batch_user_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.user_ids)
            batch_item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.item_ids)

            temp_sum = tf.reduce_sum(tf.multiply(batch_user_embeddings, batch_item_embeddings), axis=1)

            bias = tf.add(batch_user_bias, batch_item_bias)
            bias = tf.add(bias, self.global_bias)

            predictor = tf.add(bias, temp_sum)
            self.pred = tf.sigmoid(predictor, name='predictions')

    def _build_loss(self):
        with tf.name_scope('loss'):
            base_loss = tf.losses.log_loss(predictions=self.pred, labels=self.ratings)
            l2_loss = self.l2_weight * sum(map(tf.nn.l2_loss, [self.user_embeddings, self.item_embeddings]))
            self.loss = tf.add(base_loss, l2_loss)

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)
