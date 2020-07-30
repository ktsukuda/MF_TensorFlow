import configparser
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data
from MF import MF


def train(model, data_splitter, batch_size, config):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.getint('MODEL', 'epoch')):
            start = 0
            total_loss = 0
            train_data = data_splitter.make_train_data(config.getint('MODEL', 'n_negative'))
            np.random.shuffle(train_data)
            while start < len(train_data):
                _, loss = model.train(
                    sess, get_feed_dict(model, train_data, start, start + batch_size))
                start += batch_size
                total_loss += loss
            print('[Epoch {}] Loss = {:.2f}'.format(epoch, total_loss))


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.user_ids] = train_data[start:end, 0]
    feed_dict[model.item_ids] = train_data[start:end, 1]
    feed_dict[model.ratings] = train_data[start:end, 2]
    return feed_dict


def main():
    config = configparser.ConfigParser()
    config.read('MF_TensorFlow/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')
    test_data = data_splitter.make_evaluation_data('test')

    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                for l2_weight in map(float, config['MODEL']['l2_weight'].split()):
                    print('batch_size = {}, lr = {}, latent_dim = {}, l2_weight = {}'.format(
                        batch_size, lr, latent_dim, l2_weight))
                    model = MF(data_splitter.n_user, data_splitter.n_item, lr, latent_dim, l2_weight)
                    train(model, data_splitter, batch_size, config)


if __name__ == "__main__":
    main()
