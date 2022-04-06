import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm.modeling_xlm import XLMModel
from torch.autograd import Variable
import sys

sys.path.append("./src/SSDM/")
import model_utils
from Dependency_paring import probe
from Dependency_paring.loss import L1DepthLoss, L1DistanceLoss
from decorators import auto_init_args, auto_init_pytorch
from von_mises_fisher import VonMisesFisher

MODEL_CLASSES = {
    'mbert': BertModel,
    'xlm': XLMModel,
    'xlmr': XLMRobertaModel
}

MAX_LEN_LIST = {
    'mbert': 180,
    'xlm': 146,
    'xlmr': 166
}


def word_avg(input_vecs, mask):
    sum_vecs = (input_vecs * mask.unsqueeze(-1)).sum(1)
    avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
    return avg_vecs


class bag_of_words(nn.Module):
    def __init__(self, ysize, zsize, mlp_layer, hidden_size,
                 vocab_size, dropout, *args, **kwargs):
        super(bag_of_words, self).__init__()
        self.hid2vocab = model_utils.get_mlp(
            ysize + zsize,
            hidden_size,
            vocab_size,
            mlp_layer,
            dropout)

    def forward(self, yvecs, zvecs, tgts, tgts_mask):
        input_vecs = torch.cat([yvecs, zvecs], -1)
        logits = F.log_softmax(self.hid2vocab(input_vecs), -1)
        if torch.cuda.is_available():
            logits.cuda()
        return -(torch.sum(logits * tgts, 1) / tgts.sum(1)).mean()


class MultiDisModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, experiment):
        super(MultiDisModel, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.margin = self.expe.config.m
        self.use_cuda = self.expe.config.use_cuda
        self.MAX_LEN = MAX_LEN_LIST[self.experiment.config.ml_type]

        y_out_size = embed_dim
        z_out_size = embed_dim

        self.mean1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.logvar1 = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=1,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)

        self.mean2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.logvar2 = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.decode = bag_of_words(
            ysize=self.expe.config.ysize,
            zsize=self.expe.config.zsize,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            hidden_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size)

        self.pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=self.MAX_LEN,
            dropout=self.expe.config.dp)

        self.classifier = nn.Linear(self.expe.config.zsize, 2)

        self.probe_depth = probe.OneWordPSDProbe(use_cuda=self.expe.config.use_cuda, model_dim=self.expe.config.edim)
        self.probe_distance = probe.TwoWordPSDProbe(use_cuda=self.expe.config.use_cuda, model_dim=self.expe.config.edim)
        self.L1DepthLoss = L1DepthLoss()
        self.L1DistanceLoss = L1DistanceLoss()

    def pos_loss(self, mask, vecs):
        mask = mask.float()
        batch_size, seq_len = mask.size()
        # batch size x seq len x MAX LEN
        logits = self.pos_decode(vecs)
        if self.MAX_LEN - seq_len:
            padded = torch.zeros(batch_size, self.MAX_LEN - seq_len)
            if torch.cuda.is_available():
                padded = padded.cuda()
            new_mask = 1 - torch.cat([mask, self.to_var(padded)], -1)
        else:
            new_mask = 1 - mask
        new_mask = new_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(new_mask.data.bool().byte(), -bool('inf'))
        loss = F.softmax(logits, -1)[:, np.arange(int(seq_len)),
               np.arange(int(seq_len))]
        loss = -(loss + self.eps).log() * mask

        loss = loss.sum(-1) / mask.sum(1)
        return loss.mean()

    def sample_gaussian(self, mean, logvar):
        sample = mean + torch.exp(0.5 * logvar) * \
                 Variable(logvar.data.new(logvar.size()).normal_())
        return sample

    def to_var(self, inputs):
        if self.use_cuda:
            if isinstance(inputs, Variable):
                inputs = inputs.cuda()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs).cuda()
                return Variable(inputs, volatile=self.volatile)
        else:
            if isinstance(inputs, Variable):
                inputs = inputs.cpu()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs, volatile=self.volatile)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and
                                        inputs_.size else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def save(self, dev_avg, dev_perf, test_avg,
             test_perf, epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_perf": dev_perf,
            "test_perf": test_perf,
            "dev_avg": dev_avg,
            "test_avg": test_avg,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict):
        self.load_state_dict(checkpointed_state_dict)
        self.expe.log.info("model loaded!")

    @property
    def volatile(self):
        return not self.training


class vgvae(MultiDisModel):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, embed_dim, experiment, tags_vocab_size):
        super(vgvae, self).__init__(vocab_size, embed_dim, experiment)
        pre_trained_model = MODEL_CLASSES[self.experiment.config.ml_type]
        self.transformer_model = pre_trained_model.from_pretrained(self.experiment.config.ml_token)
        self.transformer_model.eval()

        self.embed_dim = embed_dim
        self.fc = nn.Linear(self.experiment.config.edim + self.experiment.config.zsize, tags_vocab_size)

    def sent2param(self, sent, mask):
        with torch.no_grad():
            encoded_layers = self.transformer_model(sent.long())
            input_vecs = encoded_layers[0]
        yvecs = input_vecs
        zvecs = input_vecs

        mean = self.mean1(yvecs)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        mean = word_avg(mean, mask)

        logvar = self.logvar1(yvecs)
        var = F.softplus(logvar) + 100
        var = word_avg(var, mask)

        mean2 = self.mean2(zvecs)
        mean2 = word_avg(mean2, mask)

        logvar2 = self.logvar2(zvecs)
        logvar2 = word_avg(logvar2, mask)

        return zvecs, mean, var, mean2, logvar2

    def forward(self, sent1, mask1, sent2, mask2, tgt1,
                tgt_mask1, tgt2, tgt_mask2,
                neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                gtemp, tree_tags, y, use_margin, true_it):
        global ploss1, ploss2, ploss3, ploss4
        self.train()

        sent1, mask1, sent2, mask2, tgt1, \
        tgt_mask1, tgt2, tgt_mask2, neg_sent1, \
        neg_mask1, ntgt1, ntgt_mask1, neg_sent2, \
        neg_mask2, ntgt2, ntgt_mask2 = \
            self.to_vars(sent1, mask1, sent2, mask2, tgt1,
                         tgt_mask1, tgt2, tgt_mask2,
                         neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                         neg_sent2, neg_mask2, ntgt2, ntgt_mask2)

        s1_vecs, sent1_mean, sent1_var, sent1_mean2, sent1_logvar2 = self.sent2param(sent1, mask1)
        s2_vecs, sent2_mean, sent2_var, sent2_mean2, sent2_logvar2 = self.sent2param(sent2, mask2)

        sent1_dist = VonMisesFisher(sent1_mean, sent1_var)
        sent2_dist = VonMisesFisher(sent2_mean, sent2_var)

        sent1_syntax = self.sample_gaussian(sent1_mean2, sent1_logvar2)
        sent2_syntax = self.sample_gaussian(sent2_mean2, sent2_logvar2)

        sent1_semantic = sent1_dist.rsample()
        sent2_semantic = sent2_dist.rsample()

        logloss1 = self.decode(
            sent1_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss2 = self.decode(
            sent2_semantic, sent2_syntax, tgt2, tgt_mask2)

        logloss3 = self.decode(
            sent2_semantic, sent1_syntax, tgt1, tgt_mask1)
        logloss4 = self.decode(
            sent1_semantic, sent2_syntax, tgt2, tgt_mask2)

        sent1_kl = model_utils.gauss_kl_div(
            sent1_mean2, sent1_logvar2,
            eps=self.eps).mean()
        sent2_kl = model_utils.gauss_kl_div(
            sent2_mean2, sent2_logvar2,
            eps=self.eps).mean()

        sdl = torch.zeros_like(sent1_kl)

        if use_margin and true_it > 300:
            n1_vecs, nsent1_mean, nsent1_var, nsent1_mean2, nsent1_logvar2 = \
                self.sent2param(neg_sent1, neg_mask1)
            n2_vecs, nsent2_mean, nsent2_var, nsent2_mean2, nsent2_logvar2 = \
                self.sent2param(neg_sent2, neg_mask2)

            sent_cos_pos = F.cosine_similarity(sent1_mean, sent2_mean)

            sent1_cos_neg = F.cosine_similarity(sent1_mean, nsent1_mean)
            sent2_cos_neg = F.cosine_similarity(sent2_mean, nsent2_mean)

            sdl += F.relu(self.margin - sent_cos_pos + sent1_cos_neg) + \
                    F.relu(self.margin - sent_cos_pos + sent2_cos_neg)

            #dist += dist_.mean()

        vkl = sent1_dist.kl_div().mean() + sent2_dist.kl_div().mean()

        gkl = sent1_kl + sent2_kl

        rec_logloss = logloss1 + logloss2

        para_logloss = logloss3 + logloss4

        if self.expe.config.pratio:
            s1_vecs = torch.cat(
                [s1_vecs, sent1_syntax.unsqueeze(1).expand(-1, s1_vecs.size(1), -1)], -1)
            s2_vecs = torch.cat(
                [s2_vecs, sent2_syntax.unsqueeze(1).expand(-1, s2_vecs.size(1), -1)], -1)
            ploss1 = self.pos_loss(mask1, s1_vecs)
            ploss2 = self.pos_loss(mask2, s2_vecs)

            ploss = ploss1 + ploss2
        else:
            ploss = torch.zeros_like(gkl)

        loss = self.expe.config.lratio * rec_logloss + \
               self.expe.config.plratio * para_logloss + \
               vtemp * vkl + gtemp * gkl + \
               self.expe.config.pratio * ploss + sdl

        stl = torch.zeros_like(sent1_kl)
        if self.expe.config.dpratio:
            # sentence1
            if torch.cuda.is_available():
                for i in range(len(tree_tags)):
                    tree_tags[i] = tree_tags[i].cuda()
            sentence1_length = s1_vecs.size()[1]
            predict_dep1 = self.probe_depth(s1_vecs)
            loss11, _ = self.L1DepthLoss(predict_dep1, torch.from_numpy(tree_tags[0]), sentence1_length)
            predict_dis1 = self.probe_distance(s1_vecs)
            loss12, _ = self.L1DistanceLoss(predict_dis1, torch.from_numpy(tree_tags[2]), sentence1_length)

            # sentence2
            sentence2_length = s2_vecs.size()[1]
            predict_dep2 = self.probe_depth(s2_vecs)
            loss21, _ = self.L1DepthLoss(predict_dep2, torch.from_numpy(tree_tags[1]), sentence2_length)
            predict_dis2 = self.probe_distance(s2_vecs)
            loss22, _ = self.L1DistanceLoss(predict_dis2, torch.from_numpy(tree_tags[3]), sentence2_length)
            stl += loss11.mean() + loss12.mean() + loss21.mean() + loss22.mean()
            loss += self.expe.config.dpratio * stl

        pos_loss = torch.zeros_like(sent1_kl)
        if self.expe.config.posratio:
            # sentence1
            logits = self.fc(s1_vecs)
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y_1 = torch.from_numpy(y[0])
            # print(y.size())
            y_1 = y_1.view(-1)  # (N*T,)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            if torch.cuda.is_available():
                pos_loss += criterion(logits, y_1.cuda())
            else:
                pos_loss += criterion(logits, y_1)

            # sentence2
            logits = self.fc(s2_vecs)
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y_2 = torch.from_numpy(y[1])
            y_2 = y_2.view(-1)  # (N*T,)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            if torch.cuda.is_available():
                pos_loss += self.expe.config.posratio * criterion(logits, y_2.cuda())  # this pos loss is for low-source languages
            else:
                pos_loss += self.expe.config.posratio * criterion(logits, y_2)
            loss += pos_loss

        return loss, vkl, gkl, rec_logloss, para_logloss, ploss, sdl, pos_loss, stl

    def score(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)
        if self.expe.config.use_cuda:
            sent1 = sent1.cuda()
            mask1 = mask1.cuda()
            sent2 = sent2.cuda()
            mask2 = mask2.cuda()
        sent1_vecs = self.transformer_model(sent1.long())
        sent2_vecs = self.transformer_model(sent2.long())
        sent1_vec = self.mean1(sent1_vecs[0])
        sent2_vec = self.mean1(sent2_vecs[0])
        sent1_vec = sent1_vec / sent1_vec.norm(dim=-1, keepdim=True)
        sent2_vec = sent2_vec / sent2_vec.norm(dim=-1, keepdim=True)
        sent1_vec = word_avg(sent1_vec, mask1)
        sent2_vec = word_avg(sent2_vec, mask2)

        return model_utils.pariwise_cosine_similarity(
            sent1_vec, sent2_vec).data.cpu().numpy()

    def pred(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)
        if self.expe.config.use_cuda:
            sent1 = sent1.cuda()
            mask1 = mask1.cuda()
            sent2 = sent2.cuda()
            mask2 = mask2.cuda()

        sent1_vecs = self.transformer_model(sent1.long())
        sent2_vecs = self.transformer_model(sent2.long())
        sent1_mean = self.mean1(sent1_vecs[0])
        sent1_mean = sent1_mean / sent1_mean.norm(dim=-1, keepdim=True)
        sent1_mean = word_avg(sent1_mean, mask1)

        sent2_mean = self.mean1(sent2_vecs[0])
        sent2_mean = sent2_mean / sent2_mean.norm(dim=-1, keepdim=True)
        sent2_mean = word_avg(sent2_mean, mask2)

        sent_cos_pos = F.cosine_similarity(sent1_mean, sent2_mean)
        return sent_cos_pos.data.cpu().numpy()

    def predz(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)
        if self.expe.config.use_cuda:
            sent1 = sent1.cuda()
            mask1 = mask1.cuda()
            sent2 = sent2.cuda()
            mask2 = mask2.cuda()
        sent1_vecs = self.transformer_model(sent1.long())
        sent2_vecs = self.transformer_model(sent2.long())
        sent1_mean = self.mean2(sent1_vecs[0])
        sent1_mean = sent1_mean / sent1_mean.norm(dim=-1, keepdim=True)
        sent1_mean2 = word_avg(sent1_mean, mask1)

        sent2_mean = self.mean2(sent2_vecs[0])
        sent2_mean = sent2_mean / sent2_mean.norm(dim=-1, keepdim=True)
        sent2_mean2 = word_avg(sent2_mean, mask2)

        sent_cos_pos = F.cosine_similarity(sent1_mean2, sent2_mean2)
        return sent_cos_pos.data.cpu().numpy()
