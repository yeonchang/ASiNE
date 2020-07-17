# -*- coding: utf-8 -*-
import os, sys, time, datetime, collections, pickle
import argparse
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import tqdm
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import generator
import discriminator
import utils

class ASiNE(object):
    def __init__(self, config): 
        print("Start initializing... ")
        self.config = config
        self.pos_graph, self.neg_graph, self.n_node, self.pos_roots, self.neg_roots = \
                                            utils.read_edges(self.config.train_filename, self.config.test_filename)
        
        self.train_graph = utils.read_edges_from_file(self.config.train_filename)
        self.test_graph = utils.read_edges_from_file(self.config.test_filename)

        init_delta_D = 0.05
        init_delta_G = 0.05
        self.node_emb_init_d = tf.Variable(tf.random_uniform([self.n_node, self.config.n_emb], minval=-init_delta_D, maxval=init_delta_D, dtype=tf.float32))
        self.node_emb_init_g = tf.Variable(tf.random_uniform([self.n_node, self.config.n_emb], minval=-init_delta_G, maxval=init_delta_G, dtype=tf.float32))

        # construct or read BFS-trees
        self.pos_partition_trees = {}
        self.neg_partition_trees = {}

        self.pos_discriminator = None
        self.neg_discriminator = None
        self.pos_generator = None
        self.neg_generator = None
        self.build_generator()
        self.build_discriminator()

        self.latest_checkpoint = tf.train.latest_checkpoint(self.config.model_log)
        self.saver = tf.train.Saver()

        self.config_tf = tf.ConfigProto()
        self.config_tf.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config_tf)
        self.sess.run(self.init_op)
        print("End initializing.")


    def construct_trees(self, graph, nodes):
        """
        construct BFS trees with nodes
        config: (sub) graph, nodes according to the graph
        returns: trees (dictionary; node_id: [parent, child_0, child_1, ...])
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)

                for sub_node in graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees


    def build_generator(self):
        # building generator
        with tf.variable_scope("generator") as generator_scope:
            self.pos_generator = generator.Generator(n_node=self.n_node, 
                                      node_emb_init=self.node_emb_init_g, 
                                      positive=True, config=self.config)
            generator_scope.reuse_variables()
            self.neg_generator = generator.Generator(n_node=self.n_node, 
                                      node_emb_init=self.node_emb_init_g, 
                                      positive=False, config=self.config)


    def build_discriminator(self):
        # building discriminator
        with tf.variable_scope("discriminator") as discriminator_scope:
            self.pos_discriminator = discriminator.Discriminator(
                                        n_node=self.n_node, 
                                        node_emb_init=self.node_emb_init_d, 
                                        positive=True, config=self.config)
            discriminator_scope.reuse_variables()
            self.neg_discriminator = discriminator.Discriminator(
                                        n_node=self.n_node, 
                                        node_emb_init=self.node_emb_init_d, 
                                        positive=False, config=self.config)


    def train(self):
        print("Start training...")
        checkpoint = tf.train.get_checkpoint_state(self.config.model_log)
        if checkpoint and checkpoint.model_checkpoint_path and self.config.load_model:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        start_time = time.time()
        ss_times = []
        mean_interval = []

        for epoch in range(self.config.n_epochs):
            print("Epoch {}".format(str(epoch)))
            ss_times.append(time.time())

            root_parts = int(len(self.pos_roots) / self.config.n_node_subsets)
            np.random.shuffle(self.pos_roots)
            pos_roots_parted = list(utils.divide_chunks(self.pos_roots, root_parts))

            if epoch > 0 and epoch % self.config.save_steps == 0:   # save the model
                self.saver.save(self.sess, self.config.model_log + ".model.checkpoint")

            print("\t Performing adversarial learning in positive graph...")
            for roots in pos_roots_parted:
                self.pos_partition_trees = self.construct_trees(self.pos_graph, roots)
                pos_dis_vars, pos_gen_vars = self.prepare_data_positive(roots)

                # positive discriminator step
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                for d_epoch in range(self.config.n_epochs_dis):
                    center_nodes, neighbor_nodes, labels = pos_dis_vars[0], pos_dis_vars[1], pos_dis_vars[2]

                    train_size = len(center_nodes)
                    start_list = list(range(0, train_size, self.config.batch_size_dis))
                    for start in start_list:
                        end = start + self.config.batch_size_dis
                        self.sess.run(self.pos_discriminator.d_updates,
                                      feed_dict={self.pos_discriminator.node_id: np.array(center_nodes[start:end]),
                                                 self.pos_discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                                 self.pos_discriminator.label: np.array(labels[start:end])})

                # positive generator step
                for g_epoch in range(self.config.n_epochs_gen):
                    node_1, node_2, reward = pos_gen_vars[0], pos_gen_vars[1], pos_gen_vars[2]

                    train_size = len(node_1)
                    start_list = list(range(0, train_size, self.config.batch_size_gen))
                    # np.random.shuffle(start_list)
                    for start in start_list:
                        end = start + self.config.batch_size_gen
                        self.sess.run(self.pos_generator.g_updates,
                                      feed_dict={self.pos_generator.node_id: np.array(node_1[start:end]),
                                                 self.pos_generator.node_neighbor_id: np.array(node_2[start:end]),
                                                 self.pos_generator.reward: np.array(reward[start:end])})
            
            root_parts = int(len(self.neg_roots) / self.config.n_node_subsets)
            np.random.shuffle(self.neg_roots)
            neg_roots_parted = list(utils.divide_chunks(self.neg_roots, root_parts))

            print("\t Performing adversarial learning in negative graph...")
            for roots in neg_roots_parted:
                self.neg_partition_trees = self.construct_trees(self.neg_graph, roots)
                neg_dis_vars, neg_gen_vars, pos_dis_vars, pos_gen_vars = self.prepare_data_negative(roots)

                # negative discriminator step 
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                for d_epoch in range(self.config.n_epochs_dis):
                    center_nodes, neighbor_nodes, labels = neg_dis_vars[0], neg_dis_vars[1], neg_dis_vars[2]

                    train_size = len(center_nodes)
                    start_list = list(range(0, train_size, self.config.batch_size_dis))
                    np.random.shuffle(start_list)
                    for start in start_list:
                        end = start + self.config.batch_size_dis
                        self.sess.run(self.neg_discriminator.d_updates,
                                      feed_dict={self.neg_discriminator.node_id: np.array(center_nodes[start:end]),
                                                 self.neg_discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                                 self.neg_discriminator.label: np.array(labels[start:end])})

                # negative generator step
                for g_epoch in range(self.config.n_epochs_gen):
                    neg_node_1, neg_node_2, neg_reward = neg_gen_vars[0], neg_gen_vars[1], neg_gen_vars[2]

                    train_size = len(neg_node_1)
                    start_list = list(range(0, train_size, self.config.batch_size_gen))
                    np.random.shuffle(start_list)
                    for n_start in start_list:
                        end = n_start + self.config.batch_size_gen
                        self.sess.run(self.neg_generator.g_updates,
                                      feed_dict={self.neg_generator.node_id: np.array(neg_node_1[n_start:end]),
                                                 self.neg_generator.node_neighbor_id: np.array(neg_node_2[n_start:end]),
                                                 self.neg_generator.reward: np.array(neg_reward[n_start:end])})

                # positive discriminator step with fake positive pair from negative generation
                if self.config.learn_fake_pos == True:
                    center_nodes = []
                    neighbor_nodes = []
                    labels = []
                    dis_all_cnt = 0
                    for d_epoch in range(self.config.n_epochs_dis):
                        center_nodes, neighbor_nodes, labels = pos_dis_vars[0], pos_dis_vars[1], pos_dis_vars[2]

                        train_size = len(center_nodes)
                        start_list = list(range(0, train_size, self.config.batch_size_dis))
                        np.random.shuffle(start_list)
                        for start in start_list:
                            end = start + self.config.batch_size_dis
                            self.sess.run(self.pos_discriminator.d_updates,
                                          feed_dict={self.pos_discriminator.node_id: np.array(center_nodes[start:end]),
                                                     self.pos_discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                                     self.pos_discriminator.label: np.array(labels[start:end])})

                # positive generator step with fake positive pair from negative generation
                if self.config.learn_fake_pos == True:
                    for g_epoch in range(self.config.n_epochs_gen):
                        pos_node_1, pos_node_2, pos_reward = pos_gen_vars[0], pos_gen_vars[1], pos_gen_vars[2]

                        train_size = len(pos_node_1)
                        start_list = list(range(0, train_size, self.config.batch_size_gen))
                        np.random.shuffle(start_list)
                        for start in start_list:
                            end = start + self.config.batch_size_gen
                            self.sess.run(self.pos_generator.g_updates,
                                          feed_dict={self.pos_generator.node_id: np.array(pos_node_1[start:end]),
                                                     self.pos_generator.node_neighbor_id: np.array(pos_node_2[start:end]),
                                                     self.pos_generator.reward: np.array(pos_reward[start:end])})            
            self.evaluation(self, epoch)
        self.write_embeddings_to_file()
        print("Complete training")


    def prepare_data_positive(self, roots):
        dis_centers = []
        dis_neighbors = []
        dis_labels = []

        gen_pair_node_1 = []
        gen_pair_node_2 = []
        gen_paths = []

        for i in roots:
            if np.random.rand() < self.config.update_ratio:
                real = self.pos_graph[i]

                n_sample = max(len(real), self.config.n_sample_gen)
                fake, paths_from_i = self.pos_sample(i, self.pos_partition_trees[i], n_sample, for_d=True)
                if paths_from_i is None:
                    continue

                dis_centers.extend([i] * len(real))
                dis_neighbors.extend(real)
                dis_labels.extend([1] * len(real))

                dis_centers.extend([i] * len(real))
                dis_neighbors.extend(fake[:len(real)])
                dis_labels.extend([0] * len(real))

                gen_paths.extend(paths_from_i)

        node_pairs = list(map(self.get_node_pairs_from_path, gen_paths))

        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                gen_pair_node_1.append(pair[0])
                gen_pair_node_2.append(pair[1])

        gen_reward = self.sess.run(self.pos_discriminator.reward, 
                               feed_dict={self.pos_discriminator.node_id: np.array(gen_pair_node_1),
                                          self.pos_discriminator.node_neighbor_id: np.array(gen_pair_node_2)})

        return (dis_centers, dis_neighbors, dis_labels), (gen_pair_node_1, gen_pair_node_2, gen_reward)


    def prepare_data_negative(self, roots):
        neg_dis_centers = []
        neg_dis_neighbors = []
        neg_dis_labels = []

        pos_dis_centers = []
        pos_dis_neighbors = []
        pos_dis_labels = []

        neg_gen_node_pair_1 = []
        neg_gen_node_pair_2 = []

        pos_gen_node_pair_1 = []
        pos_gen_node_pair_2 = []

        gen_paths = []

        for i in roots:
            if np.random.rand() < self.config.update_ratio:
                real = self.neg_graph[i]

                n_sample = max(len(real), self.config.n_sample_gen)
                neg_fakes, pos_fakes, paths_from_i = self.neg_sample(i, self.neg_partition_trees[i], n_sample, for_d=True)
                if paths_from_i is None:
                    continue

                neg_dis_centers.extend([i] * len(real))
                neg_dis_neighbors.extend(real)
                neg_dis_labels.extend([1] * len(real))

                neg_dis_centers.extend([i] * len(real))
                neg_dis_neighbors.extend(neg_fakes[:len(real)])
                neg_dis_labels.extend([0] * len(real))

                if self.config.learn_fake_pos == True:
                    if self.pos_graph.get(i) is not None:
                        real = self.pos_graph[i]
                        n_pairs = min(len(real), len(pos_fakes))
 
                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(real[:n_pairs])
                        pos_dis_labels.extend([1] * n_pairs)

                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(pos_fakes[:n_pairs])
                        pos_dis_labels.extend([0] * n_pairs)

                gen_paths.extend(paths_from_i)

        gen_node_pairs = list(map(self.get_node_pairs_from_path, gen_paths))
        gen_node_pairs_sign = list(map(self.get_node_pairs_sign_from_path, gen_paths))

        for i_path in range(len(gen_node_pairs)): # i-th pair
            for j_pair in range(len(gen_node_pairs[i_path])):
                if gen_node_pairs_sign[i_path][j_pair] == [-1]:
                    neg_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                    neg_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])
                else:
                    if self.config.learn_fake_pos == True:
                        pos_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                        pos_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])

        gen_neg_reward = self.sess.run(self.neg_discriminator.reward, 
                               feed_dict={self.neg_discriminator.node_id: np.array(neg_gen_node_pair_1),
                                          self.neg_discriminator.node_neighbor_id: np.array(neg_gen_node_pair_2)})

        gen_pos_reward = self.sess.run(self.pos_discriminator.reward, 
                               feed_dict={self.pos_discriminator.node_id: np.array(pos_gen_node_pair_1),
                                          self.pos_discriminator.node_neighbor_id: np.array(pos_gen_node_pair_2)})


        return (neg_dis_centers, neg_dis_neighbors, neg_dis_labels), \
               (neg_gen_node_pair_1, neg_gen_node_pair_2, gen_neg_reward), \
               (pos_dis_centers, pos_dis_neighbors, pos_dis_labels), \
               (pos_gen_node_pair_1, pos_gen_node_pair_2, gen_pos_reward)


    def pos_sample(self, root, tree, sample_num, for_d):
        """
        sample nodes from positive BFS-tree
        config:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """
        fakes = []
        paths = []
        n = 0

        while len(fakes) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)

            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None

                if for_d:  # skip 1-hop nodes (positive fakes)
                    if node_neighbor == [root]:
                        return None, None

                    if root in node_neighbor:
                        node_neighbor.remove(root)

                target_score = self.sess.run(self.pos_generator.target_score, 
                                             feed_dict={self.pos_generator.target_node: np.array([current_node])})

                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]
                relevance_probability = np.nan_to_num(relevance_probability)

                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)

                if next_node == previous_node:  # terminating condition
                    fakes.append(current_node)
                    break

                previous_node = current_node
                current_node = next_node

            n = n + 1

        return fakes, paths


    def neg_sample(self, root, tree, sample_num, for_d):
        """
        sample nodes from nagetive BFS-tree (via random walk?)
        config:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """
        neg_fakes = []
        pos_fakes = []         
        paths = []
        n = 0

        while len(neg_fakes) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)

            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None, None
                if for_d:  # skip 1-hop nodes (positive neg_fakes)
                    if node_neighbor == [root]:
                        return None, None, None

                    if root in node_neighbor:
                        node_neighbor.remove(root)

                target_score = self.sess.run(self.neg_generator.target_score, 
                                             feed_dict={self.neg_generator.target_node: np.array([current_node])})

                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]
                relevance_probability = np.nan_to_num(relevance_probability)
                relevance_probability = utils.softmax(1 - (utils.softmax(relevance_probability)))

                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node

                while current_node == next_node:
                    next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node

                paths[n].append(next_node)

                if next_node == previous_node:  # terminating condition
                    if len(paths[n]) % 2 == 1:  # if len(path) is odd, sample current node
                        neg_fakes.append(current_node)
                    else: # if len(path) is even, sample next node
                        neg_fakes.append(next_node)

                    if self.config.learn_fake_pos == True:
                        if len(paths[n]) % 2 == 1: # if len(path) is odd, sample next node
                            pos_fakes.append(next_node)
                        else:
                            pos_fakes.append(current_node) # if len(path) is odd, sample current node
                    break

                previous_node = current_node
                current_node = next_node

            n = n + 1

        return neg_fakes, pos_fakes, paths


    def get_node_pairs_from_path(self, path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """
        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - self.config.window_size, 0), min(i + self.config.window_size + 1, len(path))):
                if i == j or path[i] == path[j]:
                    continue

                node = path[j]
                pairs.append([center_node, node])

        return pairs


    def get_node_pairs_sign_from_path(self, path):
        path = path[:-1] # remove duplicate walk
        pairs_sign = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - self.config.window_size, 0), min(i + self.config.window_size + 1, len(path))):
                if i == j or path[i] == path[j]:
                    continue

                if np.abs(i - j) % 2 == 1:
                    pairs_sign.append([-1])
                else:
                    pairs_sign.append([1])

        return pairs_sign


    def write_embeddings_to_file(self):
        # write embeddings of the generator and the discriminator to files
        modes = [self.pos_generator, self.pos_discriminator, self.neg_generator, self.neg_discriminator]

        for i in range(len(modes)):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(self.config.emb_filenames[i] + ".emb", "w+") as f:
                lines = [str(self.n_node) + "\t" + str(self.config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)


    @staticmethod
    def evaluation(self, epoch):
        modes = [self.pos_generator, self.pos_discriminator]
        link_method = "concatenation"

        for i in range(len(modes)):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            X_train = []; Y_train = []
            for edge in self.train_graph:
                y = edge[2] if edge[2] == 1 else 0
                emb_1 = np.array(embedding_matrix[edge[0]])
                emb_2 = np.array(embedding_matrix[edge[1]])
                link_emb = utils.aggregate_link_emb(link_method, emb_1, emb_2)
                X_train.append(link_emb)
                Y_train.append(y)

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            X_test = []; Y_test = []
            for edge in self.test_graph:
                y = edge[2] if edge[2] == 1 else 0 
                emb_1 = np.array(embedding_matrix[edge[0]])
                emb_2 = np.array(embedding_matrix[edge[1]])
                link_emb = utils.aggregate_link_emb(link_method, emb_1, emb_2)
                X_test.append(link_emb)
                Y_test.append(y)

            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

            lr = LogisticRegression(solver='lbfgs', max_iter=10000)
            lr.fit(X_train, Y_train)
            test_y_score = lr.predict_proba(X_test)[:, 1]
            test_y_pred = lr.predict(X_test)

            auc_score = roc_auc_score(Y_test, test_y_score, average="macro")
            f1_score_macro = f1_score(Y_test, test_y_pred, average="macro")
            f1_score_micro = f1_score(Y_test, test_y_pred, average="micro")
            f1_score_binary = f1_score(Y_test, test_y_pred, average="binary")

            acc_result = "{}: {} test : auc_macro {:.4f} f1_macro {:.4f} f1_micro {:.4f} f1_binary {:.4f}  ({})\n".format(
                            datetime.datetime.now().isoformat(), self.config.emb_filenames[i] + ".emb".format(epoch), auc_score, f1_score_macro, f1_score_micro, f1_score_binary, link_method)
            print(acc_result)

            with open(self.config.result_filename, mode="a+") as f:
                f.write(acc_result)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASiNE.")

    parser.add_argument("--dataset", nargs="?", default="wikirfa",
                        help="Dataset name.")
    
    parser.add_argument("--n_emb", type=int, default=128, 
                        help="Embedding size. Default is 128.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate for both module. Recommend increase lr when the data is large.")
    parser.add_argument("--window_size", type=int, default=2,
                        help="Size of window for pair generating. Default is 2.")
    parser.add_argument("--learn_fake_pos", type=bool, default=False,
                        help="Additional learning of the fake positive edges generated from negative generator. Default is False")

    parser.add_argument("--n_epochs", type=int, default=20, 
                        help="Number of epochs. Default is 70.")
    parser.add_argument("--n_epochs_gen", type=int, default=10,
                        help="Number of inner loops for the generator. Default is 10.")
    parser.add_argument("--n_epochs_dis", type=int, default=10,
                        help="Number of inner loops for the discriminator. Default is 10.")
    parser.add_argument("--n_sample_gen", type=int, default=20,
                        help="Number of samples the generator. Default is 20.")

    parser.add_argument("--batch_size_gen", type=int, default=64,
                        help="Batch size for generator. Default is 64.")
    parser.add_argument("--batch_size_dis", type=int, default=64,
                        help="Batch size for discriminator. Default is 64.")
    parser.add_argument("--n_node_subsets", type=int, default=1,
                        help="Number of subsets to divide nodes existing in the positive or negative graph for large datasets. Default is 1.")
        
    parser.add_argument("--lambda_gen", type=float, default=1e-5,
                        help="L2 loss regulation weight for generator. Default is 1e-5.")
    parser.add_argument("--lambda_dis", type=float, default=1e-5,
                        help="L2 loss regulation weight for discriminator. Default is 1e-5.")
    
    parser.add_argument("--update_ratio", type=int, default=1,
                        help="Update ratio when choose the trees. Default is 1.")
    parser.add_argument("--load_model", type=bool, default=False,
                        help="Whether loading existing model for initialization. Default is False")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Save point for model checkpoint. Default is 10.")

    args = parser.parse_args()

    args.train_filename = "./data/" + args.dataset + "/" + args.dataset + ".train"
    args.test_filename = "./data/" + args.dataset + "/" + args.dataset + ".test"

    res_fn_path = "./results/" + args.dataset + "_dim" + str(args.n_emb) + "_lr" + str(args.lr)
    args.emb_filenames = [res_fn_path + "_gen_p", res_fn_path + "_dis_p", \
                          res_fn_path + "_gen_n", res_fn_path + "_dis_n"]
    args.result_filename = res_fn_path + ".results"
    args.modes = ["gen_p", "dis_p", "gen_n", "dis_n"]
    args.model_log = "./log/"
    args.gen_interval = args.n_epochs_gen
    args.dis_interval = args.n_epochs_dis
    args.lr_gen = args.lr
    args.lr_dis = args.lr

    return args

if __name__ == "__main__": 
    config = parse_args()
    asine = ASiNE(config)
    asine.train()
