import numpy as np
import timeit
import sys
import pickle
import random
from collections import defaultdict
import os
import argparse


def nested_int_defaultdict():
    return defaultdict(int)


def nested_float_defaultdict():
    return defaultdict(float)


def compute_cond_mutual_info(fdist_x_y_cond_z, fdist_x_cond_z, fdist_y_cond_z, fdist_z):
    cmi = 0
    total_freq = float(np.sum(list(fdist_z.values())))
    for z in fdist_x_y_cond_z:
        z_freq = fdist_z[z]
        p_z = z_freq/total_freq
        pcmi = 0
        # print(z, z_freq)
        for x_y in fdist_x_y_cond_z[z].keys():
            x, y = x_y.split('\t')
            pcmi += (fdist_x_y_cond_z[z][x_y] / float(z_freq)) * (np.log2(fdist_x_y_cond_z[z][x_y]) + np.log2(z_freq) - np.log2(fdist_x_cond_z[z][x]) - np.log2(fdist_y_cond_z[z][y]))
        cmi += p_z*pcmi
    return cmi


def compute_mutual_info(fdist_x_y, fdist_x, fdist_y):
    mi = 0
    total_freq = float(np.sum(list(fdist_x_y.values())))
    for x_y in fdist_x_y:
        # if len(x_y.split(' ')) != 2:
        #     print(x_y)
        x, y = x_y.split('\t')
        mi += (fdist_x_y[x_y] / float(total_freq)) * (
                    np.log2(fdist_x_y[x_y]) + np.log2(total_freq) - np.log2(fdist_x[x]) - np.log2(fdist_y[y]))
    return mi


def compute_entropy(fdist_x):
    H = 0
    total_freq = float(np.sum(list(fdist_x.values())))
    for x in fdist_x:
        H += -(fdist_x[x]/total_freq)*(np.log2(fdist_x[x]) - np.log2(total_freq))
    return H


def compute_cmi(fdist_x_y_z, fdist_x_z, fdist_y_z, fdist_z):
    H_x_y_z = compute_entropy(fdist_x_y_z)
    H_x_z = compute_entropy(fdist_x_z)
    H_y_z = compute_entropy(fdist_y_z)
    H_z = compute_entropy(fdist_z)
    return H_x_z + H_y_z - H_x_y_z - H_z


def load_ud_corpus(path):
    # print(path)
    f = open(path, 'r')
    data = f.readlines()
    # print(len(data), 'lines')
    f.close()
    # data = [line.decode("utf-8") for line in data]
    data = [line.strip() for line in data]
    sents=[]
    tmp = []
    for line in data:
        if line == '' and tmp != []:
            sents.append(tmp)
            tmp = []
        elif line[0] == '#':
            continue
        else:
            tokens = line.split('\t')
            if tokens[0].isdigit():
                tmp.append(line.split('\t'))
    # print(len(sents), 'sentences')
    # for sent in sents[:2]:
    #     for tokens in sent:
    #         print('\t'.join(tokens))
    return sents


def load_word_cluster_maps(vocab_path, cluster_path):
    with open(vocab_path, 'r') as f:
        lines = f.readlines()
    w_list = [line.strip() for line in lines]

    with open(cluster_path, 'rb') as f:
        clusters = pickle.load(f)

    c2w = {}
    w2c = {}

    for i, w in enumerate(w_list):
        c = clusters[i]
        if not c in c2w:
            c2w[c] = []
        c2w[c].append(w)
        w2c[w] = c
    return w2c, c2w


class InfoModel:
    def __init__(self):
        self.fdist_w1 = defaultdict(int)
        self.fdist_w2 = defaultdict(int)
        self.fdist_w1_w2 = defaultdict(int)

        self.fdist_pos1 = defaultdict(int)
        self.fdist_pos2 = defaultdict(int)
        self.fdist_pos1_pos2 = defaultdict(int)

        self.fdist_lex1 = defaultdict(int)
        self.fdist_lex2 = defaultdict(int)
        self.fdist_lex1_lex2 = defaultdict(int)

        self.data = []
        self.num_of_datapoints = 0

    def get_word_mutual_info(self):
        return compute_mutual_info(self.fdist_w1_w2, self.fdist_w1, self.fdist_w2)

    def get_pos_mutual_info(self):
        return compute_mutual_info(self.fdist_pos1_pos2, self.fdist_pos1, self.fdist_pos2)

    def get_lex_mutual_info(self):
        return compute_mutual_info(self.fdist_lex1_lex2, self.fdist_lex1, self.fdist_lex2)

    def update_model_params(self, tokens):
        w1, w2, pos1, pos2, lex1, lex2 = tokens
        self.fdist_w1[w1] += 1
        self.fdist_w2[w2] += 1
        self.fdist_w1_w2[w1 + '\t' + w2] += 1

        self.fdist_pos1[pos1] += 1
        self.fdist_pos2[pos2] += 1
        self.fdist_pos1_pos2[pos1 + '\t' + pos2] += 1

        self.fdist_lex1[lex1] += 1
        self.fdist_lex2[lex2] += 1
        self.fdist_lex1_lex2[lex1+'\t'+lex2] += 1

    def update_model_params_from_conll_file(self,  fpaths, interval, w2c, mode='data'):
        if mode == 'data' or mode == 'baseline' or (mode == 'permutation' and self.data == []):
            np.random.seed(10)
            threshold = THRESHOLD
            line_count = 0
            for fpath in fpaths:
                with open(fpath, 'r') as f:
                    sent_tmp = []
                    for line in f:
                        line = line.strip()
                        if line == '' and sent_tmp != []:
                            self.update_model_params_from_sent(sent_tmp, interval, w2c, mode)
                            sent_tmp = []
                            if self.num_of_datapoints >= threshold:
                                break
                        else:
                            sent_tmp.append(line.split('\t'))
                        line_count += 1
                    if sent_tmp != []:
                        self.update_model_params_from_sent(sent_tmp, interval, w2c, mode)
                if self.num_of_datapoints >= threshold:
                    break

    def update_model_params_from_word_pairs(self, word_pairs, w2c, permuted=False):
        tokens1 = [[w1, pos1] for [w1, _, pos1, _] in word_pairs]
        tokens2 = [[w2, pos2] for [_, w2, _, pos2] in word_pairs]

        if permuted:
            np.random.shuffle(tokens1)
            np.random.shuffle(tokens2)

        for m in range(len(word_pairs)):
            w1 = tokens1[m][0]
            w2 = tokens2[m][0]
            pos1 = tokens1[m][1]
            pos2 = tokens2[m][1]
            lex1 = str(w2c[w1])
            lex2 = str(w2c[w2])
            self.update_model_params([w1, w2, pos1, pos2, lex1, lex2])
            if permuted:
                self.data.append([w1, w2, pos1, pos2])
        self.num_of_datapoints = len(word_pairs)

    def update_model_params_from_sent(self, sent, interval, w2c, mode):
        # print(' '.join([tokens[1] for tokens in sent]))
        indice = [tokens[0] for tokens in sent]
        deps = [(tokens[0], tokens[6]) for tokens in sent]
        deps_set = set([child_index + '-' + parent_index for child_index, parent_index in deps])
        baseline_set = set()

        for child_index, parent_index in deps:

            if parent_index == '0':
                continue

            w1 = sent[int(child_index) - 1][1].lower()
            w2 = sent[int(parent_index) - 1][1].lower()

            if (not w1 in w2c) or (not w2 in w2c):
                continue

            distance = int(parent_index) - int(child_index)

            if np.abs(distance) != interval:
                continue

            if mode == 'baseline':
                np.random.shuffle(indice)
                for w1_index in indice:
                    w2_index = str(int(w1_index) + distance)
                    if w1_index != child_index and (not w1_index in baseline_set) and (1 <= int(w2_index) <= len(
                            deps)) and (w1 in w2c) and (w2 in w2c) \
                            and (not w1_index + '\t' + w2_index in deps_set) \
                            and (not w2_index + '\t' + w1_index in deps_set):
                        w1 = sent[int(w1_index) - 1][1].lower()
                        w2 = sent[int(w2_index) - 1][1].lower()
                        pos1 = sent[int(w1_index) - 1][4]
                        pos2 = sent[int(w2_index) - 1][4]
                        if (not w1 in w2c) or (not w2 in w2c):
                            continue
                        lex1 = str(w2c[w1])
                        lex2 = str(w2c[w2])
                        self.update_model_params([w1, w2, pos1, pos2, lex1, lex2])
                        self.data.append([w1, w2, pos1, pos2])
                        self.num_of_datapoints += 1
                        baseline_set.add(w1_index)
                        break
            else:
                w1 = sent[int(child_index) - 1][1].lower()
                w2 = sent[int(parent_index) - 1][1].lower()
                pos1 = sent[int(child_index) - 1][4]
                pos2 = sent[int(parent_index) - 1][4]
                lex1 = str(w2c[w1])
                lex2 = str(w2c[w2])
                self.update_model_params([w1, w2, pos1, pos2, lex1, lex2])
                self.data.append([w1, w2, pos1, pos2])
                self.num_of_datapoints += 1

            if self.num_of_datapoints >= THRESHOLD:
                break


parser = argparse.ArgumentParser(description='Mutual Info Estimation')
parser.add_argument('--corpus', type=int, default=0,
                    help='index of the corpus chunk')
args = parser.parse_args()

cluster_path = 'vocab_clusters_300.pkl'

w2c, c2w = load_word_cluster_maps('vocab_sorted_list.txt', cluster_path)

corpus_paths = ['en.00.parsed.aa', 'en.00.parsed.ab', 'en.00.parsed.ac', 'en.00.parsed.ad', 'en.00.parsed.ae',
                'en.01.parsed.aa', 'en.01.parsed.ab', 'en.01.parsed.ad', 'en.01.parsed.ae', 'en.01.parsed.ag',
                'en.02.parsed.aa', 'en.02.parsed.ab', 'en.02.parsed.ac', 'en.02.parsed.ad', 'en.02.parsed.af',
                'en.02.parsed.ag', 'en.02.parsed.ah', 'en.03.parsed.aa', 'en.03.parsed.ab', 'en.03.parsed.ac',
                'en.03.parsed.ad', 'en.03.parsed.ae', 'en.03.parsed.af', 'en.03.parsed.ag', 'en.03.parsed.ah',
                'en.03.parsed.ai', 'en.04.parsed.aa', 'en.04.parsed.ab', 'en.04.parsed.ac', 'en.04.parsed.ad',
                'en.04.parsed.ae', 'en.04.parsed.af', 'en.04.parsed.ag', 'en.04.parsed.ai', 'en.05.parsed.aa',
                'en.05.parsed.ab', 'en.05.parsed.ac', 'en.05.parsed.ad', 'en.05.parsed.ae', 'en.05.parsed.af',
                'en.05.parsed.ag', 'en.05.parsed.ah', 'en.05.parsed.ai', 'en.06.parsed.aa', 'en.06.parsed.ab',
                'en.06.parsed.ac', 'en.06.parsed.ad', 'en.06.parsed.ae', 'en.06.parsed.af', 'en.06.parsed.ag',
                'en.06.parsed.ai', 'en.07.parsed.aa', 'en.07.parsed.ab', 'en.07.parsed.ac', 'en.07.parsed.ad',
                'en.07.parsed.ae', 'en.07.parsed.af', 'en.07.parsed.ag', 'en.07.parsed.ah', 'en.07.parsed.ai',
                'en.08.parsed.aa', 'en.08.parsed.ab', 'en.08.parsed.ac', 'en.08.parsed.ad', 'en.08.parsed.ae',
                'en.08.parsed.af', 'en.08.parsed.ag', 'en.08.parsed.ah', 'en.08.parsed.ai', 'en.08.parsed.aj',
                'en.09.parsed.aa', 'en.09.parsed.ab', 'en.09.parsed.ac', 'en.09.parsed.ad', 'en.09.parsed.ae',
                'en.09.parsed.af', 'en.09.parsed.ag', 'en.09.parsed.ah', 'en.09.parsed.ai', 'en.09.parsed.aj']

chunk_index = int(args.corpus)

paths = corpus_paths[chunk_index*5:(chunk_index*5+5)]
paths = [os.path.join('./parsed_corpus', path) for path in paths]
print('corpus chunk:', ' '.join(paths))

MAX_LEN = 9
THRESHOLD = 5000000

for i in range(1, 9):

    word_pairs_data = []

    for mode in ['data', 'baseline', 'permutation']:

        start = timeit.default_timer()

        model = InfoModel()

        if mode == 'permutation':
            model.update_model_params_from_word_pairs(word_pairs_data, w2c, permuted=True)
        else:
            model.update_model_params_from_conll_file(paths, i, w2c, mode)
            if mode == 'data':
                word_pairs_data = model.data

        print(mode, 'Number of data points:', np.sum(list(model.fdist_w1.values())), model.num_of_datapoints)
        # word_mi = model.get_word_mutual_info()
        pos_mi = model.get_pos_mutual_info()
        lex_mi = model.get_lex_mutual_info()
        # print(i, 'MI(w1;w2):\t\t\t', word_mi)
        print(mode, i, 'MI(pos1;pos2)\t\t\t', pos_mi)
        print(mode, i, 'MI(lex1;lex2)\t\t\t', lex_mi)

        for k in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]:
            cut = int(len(model.data)*k)
            print(mode, 'Sample size:', cut, cut)
            print(mode, i, 'MI(pos1;pos2)\t\t\t','N/A')
            print(mode, i, 'MI(lex1;lex2)\t\t\t','N/A')

        stop = timeit.default_timer()
        print('Total time:', stop - start, '\n')

        sys.stdout.flush()
