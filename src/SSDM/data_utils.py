import csv
import pickle
import string

import numpy as np
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer

from Dependency_paring import task
from decorators import auto_init_args

punc = string.punctuation

MODEL_CLASSES = {
    'mbert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlmr': XLMRobertaTokenizer
}

def read_annotated_file(path):
    originals = []
    translations = []
    z_means = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        if "QE" in path:
            data_name = 'WNT2020_QE'
            for row in reader:
                originals.append(row["original"])
                translations.append(row["translation"])
                z_means.append(float(row['mean']))
        else:
            data_name = "zh_dev"
            for row in reader:
                try:
                    z_means.append(float(row['score'].strip()))
                    originals.append(row["text_a"])
                    translations.append(row["text_b"])
                except ValueError:
                    print(row)
    return {'data_name': data_name, 'original': originals, 'translation': translations, 'z_mean': z_means}


class data_holder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab):
        self.inv_vocab = {i: w for w, i in vocab.items()}


class data_processor:
    @auto_init_args
    def __init__(self, dp_1, dp_2, pos_path1, pos_path2, pos_id, experiment):
        self.expe = experiment
        self.dp1 = dp_1
        self.dp2 = dp_2
        self.pos_path1 = pos_path1
        self.pos_path2 = pos_path2
        self.pos_id = pos_id
        self.eval_path = self.expe.config.eval_file

    def process(self):
        vocab = self._build_pretrain_vocab(self.expe.config.ml_type)
        if self.expe.config.ml_type == "xlm":
            v = vocab.decoder
        else:
            v = vocab.vocab
        self.expe.log.info("vocab size: {}".format(len(v)))

        train_data = self._data_to_idx_dp(vocab, dp1=self.dp1, dp2=self.dp2, pos1=self.pos_path1, pos2=self.pos_path2, id=self.pos_id)

        if self.eval_path is not None:
            # read Chinese STS data
            new_data = read_annotated_file(self.eval_path[0])
            data_idx = self._data_to_idx([new_data['original'], new_data['translation']], vocab)
            dev_data = {new_data['data_name']: {}}
            dev_data[new_data['data_name']]['1'] = [data_idx[1], data_idx[0], new_data['z_mean']]

            test_data = {'zh': {}}
            new_data = read_annotated_file(self.eval_path[1])
            data_idx = self._data_to_idx([new_data['original'], new_data['translation']], vocab)
            test_data['zh']['1'] = [data_idx[1], data_idx[0], new_data['z_mean']]

            data = data_holder(
                train_data=train_data,
                dev_data=dev_data,
                test_data=test_data,
                vocab=v)
        else:
            data = data_holder(
                train_data=train_data,
                dev_data=None,
                test_data=None,
                vocab=v)

        return data, vocab

    def _data_to_idx(self, data, vocab):
        idx_pair1 = []
        idx_pair2 = []
        for d1, d2 in zip(*data):
            if isinstance(d2, str):
                s2 = vocab.convert_tokens_to_ids(vocab.tokenize(d2))
            else:
                s2 = vocab.convert_tokens_to_ids(vocab.tokenize(' '.join(d2)))
            idx_pair2.append(s2)
            if isinstance(d1, str):
                s1 = vocab.convert_tokens_to_ids(vocab.tokenize(d1))
            else:
                s1 = vocab.convert_tokens_to_ids(vocab.tokenize(' '.join(d1)))
            idx_pair1.append(s1)
        return np.array(idx_pair1), np.array(idx_pair2)

    def _data_to_idx_dp(self, vocab, dp1, dp2, pos1, pos2, id):
        idx_pair1 = []
        idx_pair2 = []

        dp1_depth_list = []
        dp1_dictance_list = []
        dp2_depth_list = []
        dp2_dictance_list = []
        depth_get = task.ParseDepthTask
        distance_get = task.ParseDistanceTask
        for d1 in dp1:
            d1_dep, sentence1 = depth_get.labels(d1)
            s1, d1_dep = self.load_depth_tag(sentence1, d1_dep, vocab)
            dp1_depth_list.append(d1_dep)
            d1_dis = distance_get.labels(d1)
            d1_dis = self.load_distance_tag(sentence1, d1_dis, vocab)
            dp1_dictance_list.append(d1_dis)
            idx_pair1.append(s1)
        for d2 in dp2:
            d2_dep, sentence2 = depth_get.labels(d2)
            s2, d2_dep = self.load_depth_tag(sentence2, d2_dep, vocab)
            dp2_depth_list.append(d2_dep)
            d2_dis = distance_get.labels(d2)
            d2_dis = self.load_distance_tag(sentence2, d2_dis, vocab)
            dp2_dictance_list.append(d2_dis)
            idx_pair2.append(s2)

        Y1 = []
        Y2 = []
        y1_max = 0
        y2_max = 0
        for s_p1, s_p2 in zip(pos1, pos2):
            words1 = [word_pos[0].lower() for word_pos in s_p1]
            tags1 = [word_pos[1] for word_pos in s_p1]
            _, Y_1 = self.load_pos_tag_to_idx(words1, tags1, vocab, id)
            Y1.append(Y_1)
            if len(Y_1) > y1_max:
                y1_max = len(Y_1)
            words2 = [word_pos[0].lower() for word_pos in s_p2]
            tags2 = [word_pos[1] for word_pos in s_p2]
            _, Y_2 = self.load_pos_tag_to_idx(words2, tags2, vocab, id)
            Y2.append(Y_2)
            if len(Y_2) > y2_max:
                y2_max = len(Y_2)
        print("max_len:", max(y1_max, y2_max))

        return np.array(idx_pair1), np.array(idx_pair2), \
               (np.array(dp1_depth_list), np.array(dp1_dictance_list)), \
               (np.array(dp2_depth_list), np.array(dp2_dictance_list)),\
                (Y1, Y2)

    def _data_to_idx_pos(self, vocab, pos1, pos2, id):
        idx_pair1 = []
        idx_pair2 = []
        Y1 = []
        Y2 = []
        for s_p1, s_p2 in zip(pos1, pos2):
            words1 = [word_pos[0].lower() for word_pos in s_p1]
            tags1 = [word_pos[1] for word_pos in s_p1]
            idx_pair_1, Y_1 = self.load_pos_tag_to_idx(words1, tags1, vocab, id)
            idx_pair1.append(idx_pair_1)
            Y1.append(Y_1)
            words2 = [word_pos[0].lower() for word_pos in s_p2]
            tags2 = [word_pos[1] for word_pos in s_p2]
            idx_pair_2, Y_2 = self.load_pos_tag_to_idx(words2, tags2, vocab, id)
            idx_pair2.append(idx_pair_2)
            Y2.append(Y_2)
        return np.array(idx_pair1), np.array(idx_pair2), (Y1, Y2)

    def load_pos_tag_to_idx(self, words, tags, vocab, id):
        assert len(words) == len(tags)
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                is_head = [1] + [0] * (len(tokens) - 1)

                t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
                yy = [id[each] for each in t]  # (T,)
                # tokens_.append(tokens)
                x.extend(xx)
                is_heads.extend(is_head)
                y.extend(yy)
            assert len(x) == len(y) == len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y),
                                                                                                      len(is_heads))

        return x, y

    def load_depth_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [], []
        for w, t in zip(words, tags):
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [t] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        return x, y

    def load_distance_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [], []
        for w, t in zip(words, tags):
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [t] + [np.ones(t.shape) * (-1)] * (len(tokens) - 1)
                x.extend(xx)
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        return y

    def _build_pretrain_vocab(self, lm_type):
        Token = MODEL_CLASSES[lm_type]
        vocab = Token.from_pretrained(self.expe.config.ml_token)
        return vocab

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class batch_accumulator:
    def __init__(self, mega_batch, p_scramble, init_batch1, init_batch2):
        assert len(init_batch1) == len(init_batch2) == mega_batch
        self.p_scramble = p_scramble
        self.mega_batch = mega_batch
        self.pool = [init_batch1, init_batch2]

    def update(self, new_batch1, new_batch2):
        self.pool[0].pop(0)
        self.pool[1].pop(0)

        self.pool[0].append(new_batch1)
        self.pool[1].append(new_batch2)

        assert len(self.pool[0]) == len(self.pool[1]) == self.mega_batch

    def get_batch(self):
        data1 = np.concatenate(self.pool[0])
        data2 = np.concatenate(self.pool[1])

        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")

        tgt_data1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")

        tgt_data2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([1] + list(sent1) + [2]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([1] + list(sent2) + [2]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.

        return input_data1, input_mask1, tgt_data1, tgt_mask1, \
               input_data2, input_mask2, tgt_data2, tgt_mask2


class bow_accumulator(batch_accumulator):

    def __init__(self, mega_batch, p_scramble, init_batch1, init_tgt1, init_batch2, init_tgt2, vocab_size):
        assert len(init_batch1) == len(init_batch2) == mega_batch
        self.p_scramble = p_scramble
        self.mega_batch = mega_batch
        self.vocab_size = vocab_size
        self.pool = [init_batch1, init_tgt1, init_batch2, init_tgt2]

    def update(self, new_batch1, new_tgt1, new_batch2, new_tgt2):
        self.pool[0].pop(0)
        self.pool[1].pop(0)
        self.pool[2].pop(0)
        self.pool[3].pop(0)

        self.pool[0].append(new_batch1)
        self.pool[1].append(new_tgt1)
        self.pool[2].append(new_batch2)
        self.pool[3].append(new_tgt2)

        assert len(self.pool[0]) == len(self.pool[1]) == self.mega_batch

    def get_batch(self):
        data1 = np.concatenate(self.pool[0])
        data2 = np.concatenate(self.pool[2])

        tgt_data1 = np.concatenate(self.pool[1])
        tgt_data2 = np.concatenate(self.pool[3])

        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

        return input_data1, input_mask1, tgt_data1, \
               input_data2, input_mask2, tgt_data2


class minibatcher:
    @auto_init_args
    def __init__(self, data1, data2, dp1, dp2, pos, vocab_size, batch_size,
                 score_func, shuffle, mega_batch, p_scramble,
                 *args, **kwargs):
        self.data1 = data1
        self.data2 = data2
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.score_func = score_func
        self.shuffle = shuffle
        self.p_scramble = p_scramble

        self.y1 = np.array(pos[0])
        self.y2 = np.array(pos[1])
        self.dp1_dep = dp1[0]
        self.dp1_dis = dp1[1]
        self.dp2_dep = dp2[0]
        self.dp2_dis = dp2[1]
        self._reset()
        self.mega_batch = mega_batch

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.dp1_dep))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.dp1_dep), self.batch_size)]

        if self.mega_batch > 1:
            init_mega_ids = self.idx_pool[-self.mega_batch:]
            init_mega1, init_mega2, init_tgt1, init_tgt2 = [], [], [], []
            for idx in init_mega_ids:
                d1, d2 = self.data1[idx], self.data2[idx]
                init_mega1.append(d1)
                init_mega2.append(d2)
                t1 = np.zeros((len(d1), self.vocab_size)).astype("float32")
                t2 = np.zeros((len(d2), self.vocab_size)).astype("float32")
                for i, (s1, s2) in enumerate(zip(d1, d2)):
                    t1[i, :] = np.bincount(s1, minlength=self.vocab_size)
                    t2[i, :] = np.bincount(s2, minlength=self.vocab_size)
                init_tgt1.append(t1)
                init_tgt2.append(t2)
            self.mega_batcher = bow_accumulator(
                self.mega_batch, self.p_scramble, init_mega1, init_tgt1,
                init_mega2, init_tgt2, self.vocab_size)

    def _select_neg_sample(self, data, data_mask,
                           cand, cand_mask, ctgt, no_diag):
        score_matrix = self.score_func(
            data, data_mask, cand, cand_mask)

        if no_diag:
            diag_idx = np.arange(len(score_matrix))
            score_matrix[diag_idx, diag_idx] = -np.inf
        neg_idx = np.argmax(score_matrix, 1)

        neg_data = cand[neg_idx]
        neg_mask = cand_mask[neg_idx]

        tgt_data = ctgt[neg_idx]
        max_len = data.shape[1]

        neg_data_ = np.zeros((neg_data.shape[0], data.shape[1])).astype("float32")
        neg_mask_ = np.zeros((neg_data.shape[0], data.shape[1])).astype("float32")
        if neg_data.shape[1] >= max_len:
            neg_data = neg_data_ + neg_data[:, : data.shape[1]]
            neg_mask = neg_mask_ + neg_mask[:, : data.shape[1]]
        else:
            neg_data = np.pad(neg_data, ((0, 0), (0, max_len - neg_data.shape[1])), 'constant', constant_values=(0))
            neg_mask = np.pad(neg_mask, ((0, 0), (0, max_len - neg_mask.shape[1])), 'constant', constant_values=(0))

        # assert neg_mask.sum(-1).max() == max_neg_len
        return score_matrix, neg_data, neg_mask, tgt_data

    def _pad(self, data1, data2, tags_1_dep, tags_1_dis, tags_2_dep, tags_2_dis, pos_1, pos_2):
        assert len(data1) == len(data2) == len(tags_1_dep) == len(tags_2_dep) == len(pos_1) == len(pos_2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])
        max_len3 = max([len(p) for p in pos_1])
        max_len4 = max([len(p) for p in pos_2])
        max_len = max([max_len1, max_len2, max_len3, max_len4])

        input_data1 = \
            np.zeros((len(data1), max_len)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), self.vocab_size)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), self.vocab_size)).astype("float32")

        input_d11 = \
            np.ones((len(tags_1_dep), max_len)).astype("int64") * (-1)
        input_d21 = \
            np.zeros((len(tags_2_dep), max_len)).astype("int64") * (-1)
        input_d12 = \
            np.ones((len(tags_1_dis), max_len, max_len)).astype("int64")
        input_d22 = \
            np.zeros((len(tags_2_dis), max_len, max_len)).astype("int64")

        input_data31 = \
            np.zeros((len(pos_1), max_len)).astype("int64")
        input_data41 = \
            np.zeros((len(pos_2), max_len)).astype("int64")

        for i, (sent1, sent2, dp11, dp12, dp21, dp22, Pos1, Pos2) in enumerate(
                zip(data1, data2, tags_1_dep, tags_1_dis, tags_2_dep, tags_2_dis, pos_1, pos_2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :] = np.bincount(sent1, minlength=self.vocab_size)

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :] = np.bincount(sent2, minlength=self.vocab_size)

            input_d11[i, :len(dp11)] = \
                np.asarray(list(dp11)).astype("int64")
            input_d21[i, :len(dp21)] = \
                np.asarray(list(dp21)).astype("int64")
            dp12 = np.array(dp12)
            dp12 = np.pad(dp12, pad_width=((0, max_len - dp12.shape[0]), (0, max_len - dp12.shape[1])), mode="constant",
                          constant_values=(-1))
            input_d12[i] = np.asarray(list(dp12)).astype("int64")
            dp22 = np.array(dp22)
            dp22 = np.pad(dp22, pad_width=((0, max_len - dp22.shape[0]), (0, max_len - dp22.shape[1])), mode="constant",
                          constant_values=(-1))
            input_d22[i] = np.asarray(list(dp22)).astype("int64")

            input_data31[i, :len(Pos1)] = \
                np.asarray(list(Pos1)).astype("int64")
            input_data41[i, :len(Pos2)] = \
                np.asarray(list(Pos2)).astype("int64")

        input_data3 = (input_d11, input_d21, input_d12, input_d22)

        if self.mega_batch > 1:
            cand1, cand_mask1, ctgt1, \
            cand2, cand_mask2, ctgt2 = \
                self.mega_batcher.get_batch()

            _, neg_data1, neg_mask1, ntgt1 = \
                self._select_neg_sample(
                    input_data1, input_mask1, cand2,
                    cand_mask2, ctgt2, False)
            _, neg_data2, neg_mask2, ntgt2 = \
                self._select_neg_sample(
                    input_data2, input_mask2, cand1,
                    cand_mask1, ctgt1, False)
            self.mega_batcher.update(data1, tgt_data1, data2, tgt_data2)

            return [input_data1, input_mask1.astype(np.bool), input_data2, input_mask2.astype(np.bool),
                    tgt_data1, tgt_data1.astype(np.bool), tgt_data2, tgt_data2.astype(np.bool),
                    neg_data1, neg_mask1.astype(np.bool), ntgt1, ntgt1.astype(np.bool),
                    neg_data2, neg_mask2.astype(np.bool), ntgt2, ntgt2.astype(np.bool), input_data3, (input_data31, input_data41)]

        return [input_data1, input_mask1.astype(np.bool), input_data2, input_mask2.astype(np.bool),
                tgt_data1, tgt_data1.astype(np.bool), tgt_data2, tgt_data2.astype(np.bool),
                None, None, None, None,
                None, None, None, None, input_data3, (input_data31, input_data41)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        tags_1_dep, tags_1_dis, tags_2_dep, tags_2_dis = self.dp1_dep[idx], self.dp1_dis[idx], self.dp2_dep[idx], \
                                                         self.dp2_dis[idx]
        data1, data2 = self.data1[idx], self.data2[idx]
        tags1, tags2 = self.y1[idx], self.y2[idx]
        self.pointer += 1
        return self._pad(data1, data2, tags_1_dep, tags_1_dis, tags_2_dep, tags_2_dis, tags1, tags2) + [idx]


class pre_minibatcher:
    @auto_init_args
    def __init__(self, data1, data2, batch_size, score_func,
                 shuffle, mega_batch, p_scramble, *args, **kwargs):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.data1))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data1), self.batch_size)]

        if self.mega_batch > 1:
            init_mega_ids = self.idx_pool[-self.mega_batch:]
            init_mega1, init_mega2 = [], []
            for idx in init_mega_ids:
                d1, d2 = self.data1[idx], self.data2[idx]
                init_mega1.append(d1)
                init_mega2.append(d2)
            self.mega_batcher = batch_accumulator(
                self.mega_batch, self.p_scramble, init_mega1, init_mega2)

    def _select_neg_sample(self, data, data_mask,
                           cand, cand_mask, ctgt, ctgt_mask, no_diag):
        score_matrix = self.score_func(
            data, data_mask, cand, cand_mask)

        if no_diag:
            diag_idx = np.arange(len(score_matrix))
            score_matrix[diag_idx, diag_idx] = -np.inf
        neg_idx = np.argmax(score_matrix, 1)

        neg_data = cand[neg_idx]
        neg_mask = cand_mask[neg_idx]

        tgt_data = ctgt[neg_idx]
        tgt_mask = ctgt_mask[neg_idx]

        max_neg_len = int(neg_mask.sum(-1).max())
        neg_data = neg_data[:, : max_neg_len]
        neg_mask = neg_mask[:, : max_neg_len]

        max_tgt_len = int(tgt_mask.sum(-1).max())
        tgt_data = tgt_data[:, : max_tgt_len]
        tgt_mask = tgt_mask[:, : max_tgt_len]

        assert neg_mask.sum(-1).max() == max_neg_len
        return score_matrix, neg_data, neg_mask, tgt_data, tgt_mask

    def _pad(self, data1, data2):
        assert len(data1) == len(data2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])
        max_len = max([max_len1, max_len2])

        input_data1 = \
            np.zeros((len(data1), max_len)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), max_len + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), max_len + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len + 2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([1] + list(sent1) + [2]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([1] + list(sent2) + [2]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.

        if self.mega_batch > 1:
            cand1, cand_mask1, ctgt1, ctgt_mask1, \
                cand2, cand_mask2, ctgt2, ctgt_mask2 = \
                self.mega_batcher.get_batch()
            _, neg_data1, neg_mask1, ntgt1, ntgt_mask1 = \
                self._select_neg_sample(
                    input_data1, input_mask1, cand2,
                    cand_mask2, ctgt2, ctgt_mask2, False)
            _, neg_data2, neg_mask2, ntgt2, ntgt_mask2 = \
                self._select_neg_sample(
                    input_data2, input_mask2, cand1,
                    cand_mask1, ctgt1, ctgt_mask1, False)
            self.mega_batcher.update(data1, data2)

            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_mask1, tgt_data2, tgt_mask2,
                    neg_data1, neg_mask1, ntgt1, ntgt_mask1,
                    neg_data2, neg_mask2, ntgt2, ntgt_mask2]
        else:
            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_mask1, tgt_data2, tgt_mask2,
                    None, None, None, None,
                    None, None, None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data1, data2 = self.data1[idx], self.data2[idx]
        self.pointer += 1
        return self._pad(data1, data2) + [idx]