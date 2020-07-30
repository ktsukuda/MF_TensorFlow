import os
import json
import math
from progressbar import ProgressBar
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
            pb = ProgressBar(1, math.ceil(len(train_data)/batch_size))
            while start < len(train_data):
                _, loss = model.train(
                    sess, get_feed_dict(model, train_data, start, start + batch_size))
                start += batch_size
                total_loss += loss
                pb.update(start // batch_size)
            hit_ratio, ndcg = evaluation.evaluate(model, sess, validation_data, config.getint('EVALUATION', 'top_k'))
            epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
            if ndcg > best_ndcg:
                tf.train.Saver().save(sess, os.path.join(result_dir, 'model'))
            print('\n[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
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


def find_best_model(config, n_user, n_item):
    best_model = None
    best_model_dir = None
    best_params = {}
    best_ndcg = 0
    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                for l2_weight in map(float, config['MODEL']['l2_weight'].split()):
                    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_weight_{}-epoch_{}-n_negative_{}-top_k_{}".format(
                        batch_size, lr, latent_dim, l2_weight, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
                    with open(os.path.join(result_dir, 'epoch_data.json')) as f:
                        ndcg = max([d['NDCG'] for d in json.load(f)])
                        if ndcg > best_ndcg:
                            best_ndcg = ndcg
                            best_params = {
                                'batch_size': batch_size, 'lr': lr, 'latent_dim': latent_dim, 'l2_weight': l2_weight}
                            best_model = MF(n_user, n_item, lr, latent_dim, l2_weight)
                            best_model_dir = result_dir
    return best_model, best_model_dir, best_params


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
                    tf.reset_default_graph()
                    model = MF(data_splitter.n_user, data_splitter.n_item, lr, latent_dim, l2_weight)
                    epoch_data = train(result_dir, model, data_splitter, validation_data, batch_size, config)
                    save_train_result(result_dir, epoch_data)

    best_model, best_model_dir, best_params = find_best_model(config, data_splitter.n_user, data_splitter.n_item)
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, os.path.join(best_model_dir, 'model'))
        hit_ratio, ndcg = evaluation.evaluate(best_model, sess, test_data, config.getint('EVALUATION', 'top_k'))
        print('---------------------------------\nBest result')
        print('batch_size = {}, lr = {}, latent_dim = {}, l2_weight = {}'.format(
            best_params['batch_size'], best_params['lr'], best_params['latent_dim'], best_params['l2_weight']))
        print('HR = {:.4f}, NDCG = {:.4f}'.format(hit_ratio, ndcg))


if __name__ == "__main__":
    main()
