import os
import json
import configparser
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data
import evaluation
from MF import MF


def train(result_dir, model, data_splitter, validation_data, batch_size, config):
    epoch_data = []
    best_ndcg = 0
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
            hit_ratio, ndcg = evaluation.evaluate(model, sess, validation_data, config.getint('EVALUATION', 'top_k'))
            epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
            if ndcg > best_ndcg:
                tf.train.Saver().save(sess, os.path.join(result_dir, 'model'))
            print('[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.user_ids] = train_data[start:end, 0]
    feed_dict[model.item_ids] = train_data[start:end, 1]
    feed_dict[model.ratings] = train_data[start:end, 2]
    return feed_dict


def save_train_result(result_dir, epoch_data):
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


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
                    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_weight_{}-epoch_{}-n_negative_{}-top_k_{}".format(
                        batch_size, lr, latent_dim, l2_weight, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
                    os.makedirs(result_dir, exist_ok=True)
                    model = MF(data_splitter.n_user, data_splitter.n_item, lr, latent_dim, l2_weight)
                    epoch_data = train(result_dir, model, data_splitter, validation_data, batch_size, config)
                    save_train_result(result_dir, epoch_data)


if __name__ == "__main__":
    main()
