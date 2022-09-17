from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import sys
import time
import logging
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf


class Trainer(object):

    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode

        # Create output-dir
        if not os.path.exists(self.config.dir_name): os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(
                self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")

        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        tf.set_random_seed(self.config[self.mode].seed)

        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net

        ## Clip Op
        self.clip_op_lambda = clip_op_lambda

        # Init TF-graph
        self.init_graph()

    def get_clip_op(self, adv_var, adv_per_var=None):
        self.clip_op = self.clip_op_lambda(adv_var)
        if adv_per_var is not None:
            self.per_clip_op = self.clip_op_lambda(adv_per_var)
        # tf.assign(adv_var, tf.clip_by_value(adv_var, 0.0, 1.0))

    def init_logger(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return tf.reduce_mean(tf.reduce_sum(pay, axis=-1))

    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return tf.reduce_sum(tf.multiply(alloc, x), axis=-1) - pay

    def get_misreports(self, x, adv_var, adv_shape):

        num_misreports = adv_shape[1]
        adv = tf.tile(tf.expand_dims(adv_var, 0), [self.config.num_agents, 1, 1, 1, 1])
        x_mis = tf.tile(x, [self.config.num_agents * num_misreports, 1, 1])
        x_r = tf.reshape(x_mis, adv_shape)
        y = x_r * (1 - self.adv_mask) + adv * self.adv_mask
        misreports = tf.reshape(y, [-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def get_permutation_inverse(self, permutation):
        permutation_reverse = np.zeros((len(permutation), len(permutation[0])), dtype=np.int)
        for i in range(len(permutation)):
            for j in range(len(permutation[0])):
                permutation_reverse[i][permutation[i][j]] = j

        return permutation_reverse

    def permute_misreport_bidder_idx(self):
        # misreports shape: #bidder x zL x bz x #bidder x #items
        # get permutation index for parallel computing

        misreport_per_idx = []
        for i, bp in enumerate(self.bidder_permutation):
            for j, ip in enumerate(self.item_permutation):
                idx = list(itertools.product(bp, bp, ip))
                misreport_per_idx.extend(idx)
        self.misreport_per_idx = misreport_per_idx

        # from  n!xm! x #bidder x #bidder x #items get permutation index
        misreport_reverse_per_idx = []
        misreport_pay_reverse_per_idx = []
        for i, bp in enumerate(self.bidder_permutation_r):
            for j, ip in enumerate(self.item_permutation_r):
                num = i * len(self.item_permutation_r) + j
                idx = np.array(list(itertools.product(bp, bp, ip)))
                idx = np.concatenate([np.ones((len(idx), 1), dtype=int) * num, idx], axis=-1)
                misreport_reverse_per_idx.append(idx)

                pay_idx = np.array(list(itertools.product(bp, bp)))
                pay_idx = np.concatenate([np.ones((len(pay_idx), 1), dtype=int) * num, pay_idx], axis=-1)
                misreport_pay_reverse_per_idx.append(pay_idx)

        self.misreport_reverse_per_idx = np.concatenate(misreport_reverse_per_idx, axis=0)
        self.misreport_pay_reverse_per_idx = np.concatenate(misreport_pay_reverse_per_idx, axis=0)

    def init_graph(self):
        x_shape = [self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size,
                     self.config.num_agents, self.config.num_items]
        adv_var_shape = [self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents,
                         self.config.num_items]
        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]

        # get permutation matrix
        self.bidder_permutation = list(itertools.permutations(np.arange(self.config.num_agents)))
        self.item_permutation = list(itertools.permutations(np.arange(self.config.num_items)))

        # get all bidders and items permuation index for parallel computing
        self.all_permutation_idx = []
        for each_b in self.bidder_permutation:
            for each_i in self.item_permutation:
                per_idx = np.array(list(itertools.product(each_b, each_i)))
                self.all_permutation_idx.append(per_idx)
        self.all_permutation_idx = np.concatenate(self.all_permutation_idx, axis=0)

        # get all bidders permuation index for parallel computing
        self.bidder_permutation_idx = []
        i = 0
        for each_b in np.array(self.bidder_permutation):
            i_idx = np.ones((len(each_b), 1), dtype=int) * i
            self.bidder_permutation_idx.append(np.concatenate([i_idx, each_b[:, None]], axis=-1))
            i = i + 1
        self.bidder_permutation_idx = np.concatenate(self.bidder_permutation_idx, axis=0)

        # get all the inverse permutation index for bidders/items.
        self.bidder_permutation_r = self.get_permutation_inverse(self.bidder_permutation)
        self.item_permutation_r = self.get_permutation_inverse(self.item_permutation)

        # get all the inverse permutation index for bidders and items.
        i = 0
        self.all_reverse_permutation_idx = []
        for each_b in self.bidder_permutation_r:
            for each_i in self.item_permutation_r:
                per_idx = np.array(list(itertools.product(each_b, each_i)))
                i_idx = np.ones((len(per_idx), 1), dtype=int) * i
                self.all_reverse_permutation_idx.append(np.concatenate([i_idx, per_idx], axis=-1))
                i = i + 1
        self.all_reverse_permutation_idx = np.concatenate(self.all_reverse_permutation_idx, axis=0)

        # get all the inverse permutation index for bidders
        self.bidder_reverse_permutation_idx = []
        i = 0
        for each_b in np.array(self.bidder_permutation_r):
            for each_i in self.item_permutation:
                i_idx = np.ones((len(each_b), 1), dtype=int) * i
                self.bidder_reverse_permutation_idx.append(np.concatenate([i_idx, each_b[:, None]], axis=-1))
                i = i + 1
        self.bidder_reverse_permutation_idx = np.concatenate(self.bidder_reverse_permutation_idx, axis=0)

        # get all the permuation index for misreport
        self.permute_misreport_bidder_idx()

        # Placeholders
        self.x = tf.placeholder(tf.float64, shape=x_shape, name='x')
        self.adv_init = tf.placeholder(tf.float64, shape=adv_var_shape, name='adv_init')

        bib_permutation = tf.transpose(self.x, [1, 2, 0])  # bidder_num x item_num x bz
        bib_permutation = tf.gather_nd(bib_permutation, self.all_permutation_idx)
        bib_permutation = tf.reshape(bib_permutation, [len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents,
                                                       self.config.num_items, -1])
        self.bib_per = tf.transpose(bib_permutation, [3, 0, 1, 2])

        # get all permuation results
        al_all, pr_all = self.net.inference(self.bib_per)
        al_all = tf.reshape(al_all,
                            [-1, len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents, self.config.num_items])
        pr_all = tf.reshape(pr_all,
                            [-1, len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents])

        al_all_trans = tf.transpose(al_all, [1, 2, 3, 0])

        # reverse the permutation to make all results have the same items and bidders location.
        alloc_per = tf.gather_nd(al_all_trans, self.all_reverse_permutation_idx)
        alloc_per = tf.reshape(alloc_per, al_all_trans.shape)
        alloc_per = tf.transpose(alloc_per, [3, 0, 1, 2])

        pr_all_trans = tf.transpose(pr_all, [1, 2, 0])
        pay_per = tf.gather_nd(pr_all_trans, self.bidder_reverse_permutation_idx)
        pay_per = tf.reshape(pay_per, pr_all_trans.shape)
        pay_per = tf.transpose(pay_per, [2, 0, 1])

        self.alloc_per_out = tf.reduce_mean(alloc_per, axis=1)
        self.pay_per_out = tf.reduce_mean(pay_per, axis=1)

        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0

        self.u_mask = np.zeros(u_shape)
        self.u_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0

        with tf.variable_scope('adv_var'):
            self.adv_var = tf.get_variable('adv_var', shape=adv_var_shape, dtype=tf.float64)
            self.adv_var_per = tf.get_variable('adv_var_per', shape=adv_var_shape, dtype=tf.float64)

        # Misreports
        x_mis, self.misreports = self.get_misreports(self.x, self.adv_var, adv_shape)
        x_mis_per, self.misreports_per = self.get_misreports(self.x, self.adv_var_per, adv_shape)

        # Misreports permutation
        # x_mis: #bidder * L * bz x #bidder x #items
        # misreports_reshape: #bidder x L x bz x #bidder x #items

        a_para_shape = [len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents, self.config.num_agents,
                        self.config.num_items, self.config[self.mode].num_misreports, -1]
        p_para_shape = [len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents, self.config.num_agents,
                        self.config[self.mode].num_misreports, -1]

        a_infer_shape = [len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents,
                         self.config[self.mode].num_misreports, -1, self.config.num_agents, self.config.num_items]
        p_infer_shape = [len(self.bidder_permutation) * len(self.item_permutation), self.config.num_agents,
                         self.config[self.mode].num_misreports, -1, self.config.num_agents]

        misreports_reshape = tf.reshape(self.misreports_per, adv_shape)  # bidder x L x bz , bidder x item
        misreports_reshape = tf.transpose(misreports_reshape, [0, 3, 4, 1, 2])  # bidder x bidder x item x L x bz

        misreports_per = tf.gather_nd(misreports_reshape, self.misreport_per_idx)

        misreports_per = tf.reshape(misreports_per, a_para_shape)
        misreports_per = tf.transpose(misreports_per, [0, 1, 4, 5, 2, 3])  # n!m! x bidder x L x bz x bidder x item

        a_mis_predict, p_mis_predict = self.net.inference(misreports_per)

        a_mis_p = tf.reshape(a_mis_predict, a_infer_shape)
        p_mis_p = tf.reshape(p_mis_predict, p_infer_shape)

        a_mis_p = tf.transpose(a_mis_p, [0, 1, 4, 5, 2, 3])  # n!m! x bidder x bidder x item x L x bz
        p_mis_p = tf.transpose(p_mis_p, [0, 1, 4, 2, 3])  # n!m! x bidder x bidder x L x bz

        # reverse permuation
        a_mis_p = tf.gather_nd(a_mis_p, self.misreport_reverse_per_idx)  # n!m! x bidder x bidder x item x L x bz
        p_mis_p = tf.gather_nd(p_mis_p, self.misreport_pay_reverse_per_idx)  # n!m! x bidder x bidder x L x bz

        a_mis_p = tf.reshape(a_mis_p, a_para_shape)
        p_mis_p = tf.reshape(p_mis_p, p_para_shape)
        a_mis_p = tf.transpose(a_mis_p, [0, 1, 4, 5, 2, 3])  # n!m! x bidder x L x bz x bidder x item
        p_mis_p = tf.transpose(p_mis_p, [0, 1, 3, 4, 2])  # n!m! x bidder x L x bz x bidder

        a_mis_per = tf.reduce_mean(a_mis_p, axis=0)  # bidder x L x bz x bidder x item
        p_mis_per = tf.reduce_mean(p_mis_p, axis=0)  # bidder x L x bz x bidder

        a_mis_per = tf.reshape(a_mis_per, [-1, self.config.num_agents, self.config.num_items])
        p_mis_per = tf.reshape(p_mis_per, [-1, self.config.num_agents])

        utility_mis_per = self.compute_utility(x_mis_per, a_mis_per, p_mis_per)
        u_mis_per = tf.reshape(utility_mis_per, u_shape) * self.u_mask
        loss_2_per = -tf.reduce_sum(u_mis_per)

        # Get mechanism for true valuation: Allocation and Payment
        self.alloc, self.pay = self.net.inference(self.x)

        # Get mechanism for misreports: Allocation and Payment
        a_mis, p_mis = self.net.inference(self.misreports)

        # Utility
        utility = self.compute_utility(self.x, self.alloc, self.pay)
        self.utility = utility
        utility_per = self.compute_utility(self.x, self.alloc_per_out, self.pay_per_out)
        self.utility_per = utility_per

        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Regret Computation
        u_mis = tf.reshape(utility_mis, u_shape) * self.u_mask
        utility_true = tf.tile(utility, [self.config.num_agents * self.config[self.mode].num_misreports, 1])
        excess_from_utility = tf.nn.relu(tf.reshape(utility_mis - utility_true, u_shape) * self.u_mask)
        rgt = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(1, 3)), axis=1)

        # Regret Computation for permuation
        utility_per_true = tf.tile(utility_per, [self.config.num_agents * self.config[self.mode].num_misreports, 1])
        excess_from_utility_per = tf.nn.relu(tf.reshape(utility_mis_per - utility_per_true, u_shape) * self.u_mask)
        rgt_per = tf.reduce_mean(tf.reduce_max(excess_from_utility_per, axis=(1, 3)), axis=1)

        # Metrics
        revenue = self.compute_rev(self.pay)
        rgt_mean = tf.reduce_mean(rgt)
        rgt_per_mean = tf.reduce_mean(rgt_per)
        irp_mean = tf.reduce_mean(tf.nn.relu(-utility))

        # Variable Lists
        alloc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
        pay_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pay')
        var_list = alloc_vars + pay_vars

        if self.mode == "train":

            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            with tf.variable_scope('lag_var'):
                self.w_rgt = tf.Variable(np.ones(self.config.num_agents).astype(np.float64) * w_rgt_init_val, 'w_rgt')

            update_rate = tf.Variable(self.config.train.update_rate, trainable=False, dtype=tf.double)
            self.increment_update_rate = update_rate.assign(update_rate + self.config.train.up_op_add)

            # Loss Functions
            rgt_penalty = update_rate * tf.reduce_sum(tf.square(rgt)) / 2.0
            lag_loss = tf.reduce_sum(self.w_rgt * rgt)

            loss_1 = -revenue + rgt_penalty + lag_loss
            loss_2 = -tf.reduce_sum(u_mis)
            loss_3 = -lag_loss

            reg_losses = tf.get_collection('reg_losses')
            if len(reg_losses) > 0:
                reg_loss_mean = tf.reduce_mean(reg_losses)
                loss_1 = loss_1 + reg_loss_mean

            learning_rate = tf.Variable(self.config.train.learning_rate, trainable=False)

            # Optimizer
            opt_1 = tf.train.AdamOptimizer(learning_rate)
            opt_2 = tf.train.AdamOptimizer(self.config.train.gd_lr)
            opt_3 = tf.train.GradientDescentOptimizer(update_rate)

            # Train ops
            self.train_op = opt_1.minimize(loss_1)
            self.train_mis_step = opt_2.minimize(loss_2, var_list=[self.adv_var])
            self.lagrange_update = opt_3.minimize(loss_3, var_list=[self.w_rgt])

            # Val ops
            val_mis_opt = tf.train.AdamOptimizer(self.config.val.gd_lr)
            self.val_mis_step = val_mis_opt.minimize(loss_2, var_list=[self.adv_var])
            self.val_mis_per_step = val_mis_opt.minimize(loss_2_per, var_list=[self.adv_var_per])

            # Reset ops
            self.reset_train_mis_opt = tf.variables_initializer(opt_2.variables())
            self.reset_val_mis_opt = tf.variables_initializer(val_mis_opt.variables())

            # Metrics
            self.train_metrics = [revenue, rgt_mean, rgt_penalty, lag_loss, loss_1, tf.reduce_mean(self.w_rgt),
                                  update_rate]
            self.train_metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss",
                                       "w_rgt_mean", "update_rate"]

            self.metrics = [revenue, rgt_mean, rgt_per_mean, rgt_penalty, lag_loss, loss_1,
                            tf.reduce_mean(self.w_rgt),
                            update_rate]
            self.metric_names = ["Revenue", "Regret", "Regret_permutation",
                                 "Reg_Loss", "Lag_Loss", "Net_Loss",
                                 "w_rgt_mean", "update_rate"]

            # Summary
            tf.summary.scalar('revenue', revenue)
            tf.summary.scalar('regret', rgt_mean)
            tf.summary.scalar('regret_per', rgt_per_mean)
            tf.summary.scalar('reg_loss', rgt_penalty)
            tf.summary.scalar('lag_loss', lag_loss)
            tf.summary.scalar('net_loss', loss_1)
            tf.summary.scalar('w_rgt_mean', tf.reduce_mean(self.w_rgt))
            if len(reg_losses) > 0: tf.summary.scalar('reg_loss', reg_loss_mean)

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=self.config.train.max_to_keep)

        elif self.mode == "test":

            loss = -tf.reduce_sum(u_mis)
            test_mis_opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss, var_list=[self.adv_var])
            self.reset_test_mis_opt = tf.variables_initializer(test_mis_opt.variables())

            # Metrics
            welfare = tf.reduce_mean(tf.reduce_sum(self.alloc * self.x, axis=(1, 2)))
            self.metrics = [revenue, rgt_mean, rgt_per_mean, irp_mean]
            self.metric_names = ["Revenue", "Regret", "Regret_permutation", "IRP"]
            self.saver = tf.train.Saver(var_list=var_list)

        # Helper ops post GD steps
        self.assign_op = tf.assign(self.adv_var, self.adv_init)
        self.assign_per_op = tf.assign(self.adv_var_per, self.adv_init)
        self.get_clip_op(self.adv_var, self.adv_var_per)

    def train(self, generator):
        """
        Runs training
        """

        self.train_gen, self.val_gen = generator

        iter = self.config.train.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(self.config.dir_name, sess.graph)

        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data(0)
            self.saver.save(sess, os.path.join(self.config.dir_name, 'model'), global_step=iter)

        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):

            # Get a mini-batch
            X, ADV, perm = next(self.train_gen.gen_func)

            if iter == 0: sess.run(self.lagrange_update, feed_dict={self.x: X})

            tic = time.time()

            # Get Best Mis-report
            sess.run(self.assign_op, feed_dict={self.adv_init: ADV})
            sess.run(self.assign_per_op, feed_dict={self.adv_init: ADV})
            for _ in range(self.config.train.gd_iter):
                sess.run(self.train_mis_step, feed_dict={self.x: X})
                sess.run(self.clip_op)
            sess.run(self.reset_train_mis_opt)

            if self.config.train.data == "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, sess.run(self.adv_var))

            # Update network params
            sess.run(self.train_op, feed_dict={self.x: X})

            if iter == 0:
                summary = sess.run(self.merged, feed_dict={self.x: X})
                train_writer.add_summary(summary, iter)

            iter += 1

            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                sess.run(self.lagrange_update, feed_dict={self.x: X})

            if iter % self.config.train.up_op_frequency == 0:
                sess.run(self.increment_update_rate)

            toc = time.time()
            time_elapsed += (toc - tic)

            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):
                self.saver.save(sess, os.path.join(self.config.dir_name, 'model'), global_step=iter)
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                summary = sess.run(self.merged, feed_dict={self.x: X})
                train_writer.add_summary(summary, iter)
                if iter > 0:
                    metric_vals = sess.run(self.metrics, feed_dict={self.x: X})
                    fmt_vals = tuple([item for tup in zip(self.metric_names, metric_vals) for item in tup])
                    log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals

                else:

                    metric_vals = sess.run(self.train_metrics, feed_dict={self.x: X})
                    fmt_vals = tuple([item for tup in zip(self.train_metric_names, metric_vals) for item in tup])
                    log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(
                        self.train_metric_names) % fmt_vals
                self.logger.info(log_str)

            if (iter % self.config.val.print_iter) == 0:
                # Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))
                # metric_tot = np.zeros(len(self.train_metric_names))
                for _ in range(self.config.val.num_batches):
                    X, ADV, _ = next(self.val_gen.gen_func)
                    sess.run(self.assign_op, feed_dict={self.adv_init: ADV})
                    sess.run(self.assign_per_op, feed_dict={self.adv_init: ADV})
                    for k in range(self.config.val.gd_iter):
                        sess.run(self.val_mis_step, feed_dict={self.x: X})
                        sess.run(self.clip_op)
                        sess.run(self.val_mis_per_step, feed_dict={self.x: X})
                        sess.run(self.per_clip_op)
                    sess.run(self.reset_val_mis_opt)
                    metric_vals = sess.run(self.metrics, feed_dict={self.x: X})
                    metric_tot += metric_vals

                metric_tot = metric_tot / self.config.val.num_batches
                fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
                log_str = "VAL-%d" % (iter) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

    def test(self, generator):
        """
        Runs test
        """

        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
        self.saver.restore(sess, model_path)

        # Test-set Stats
        time_elapsed = 0

        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert (hasattr(generator,
                            "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])

        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, ADV, perm = next(self.test_gen.gen_func)
            sess.run(self.assign_op, feed_dict={self.adv_init: ADV})
            sess.run(self.assign_per_op, feed_dict={self.adv_init: ADV})

            for k in range(self.config.test.gd_iter):
                sess.run(self.test_mis_step, feed_dict={self.x: X})
                sess.run(self.clip_op)

            sess.run(self.reset_test_mis_opt)

            metric_vals = sess.run(self.metrics, feed_dict={self.x: X})

            if self.config.test.save_output:
                A, P = sess.run([self.alloc, self.pay], feed_dict={self.x: X})
                alloc_tst[perm, :, :] = A
                pay_tst[perm, :] = P

            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([item for tup in zip(self.metric_names, metric_vals) for item in tup])
            log_str = "TEST BATCH-%d: t = %.4f" % (i, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
            self.logger.info(log_str)

        metric_tot = metric_tot / self.config.test.num_batches
        fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
        log_str = "TEST ALL-%d: t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
        self.logger.info(log_str)

        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
