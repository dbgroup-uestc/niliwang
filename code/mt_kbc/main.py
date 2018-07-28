# -*- coding: utf-8 -*-
from more_itertools import chunked
from collections import defaultdict
from more_itertools import chunked
from argparse import ArgumentParser
import sys, random, os

from models.manager import get_model
import utils.general_tool as tool

import torch

# ----------------------------------------------------------------------------

candidate_heads = defaultdict(set)
gold_heads = defaultdict(set)
candidate_tails = defaultdict(set)
gold_tails = defaultdict(set)
black_set = set()

tail_per_head = defaultdict(set)
head_per_tail = defaultdict(set)

train_data, dev_data, test_data = list(), list(), list()
trfreq = defaultdict(int)

glinks, grelations, gedges = 0, 0, 0


def initilize_dataset():
    global candidate_heads, gold_heads, candidate_heads, gold_tails
    global glinks, grelations, gedges

    # get properties of knowledge graph
    tool.trace('load train')
    grelations = dict()
    glinks = defaultdict(set)
    for line in tool.read(args.train_file):
        h, r, t = list(map(int, line.strip().split('\t')))
        grelations[(h, t)] = r
        glinks[t].add(h)
        glinks[h].add(t)
        gold_heads[(r, t)].add(h)
        gold_tails[(h, r)].add(t)
        candidate_heads[r].add(h)
        candidate_tails[r].add(t)
        tail_per_head[h].add(t)
        head_per_tail[t].add(h)

    for e in glinks:
        glinks[e] = list(glinks[e])
    for r in candidate_heads:
        candidate_heads[r] = list(candidate_heads[r])
    for r in candidate_tails:
        candidate_tails[r] = list(candidate_tails[r])
    for h in tail_per_head:
        tail_per_head[h] = len(tail_per_head[h]) + 0.0
    for t in head_per_tail:
        head_per_tail[t] = len(head_per_tail[t]) + 0.0

    tool.trace('set axiaulity')
    # switch standards setting or OOKB setting
    if args.train_file == args.auxiliary_file:
        tool.trace('standard setting, use: edges=links')
        gedges = glinks
    else:
        tool.trace('OOKB eseting, use: different edges')
        gedges = defaultdict(set)
        for line in tool.read(args.auxiliary_file):
            h, r, t = list(map(int, line.strip().split('\t')))
            grelations[(h, t)] = r
            gedges[t].add(h)
            gedges[h].add(t)
        for e in gedges:
            gedges[e] = list(gedges[e])

    global train_data, dev_data, test_data, trfreq

    # load train
    train_data = set()
    for line in open(args.train_file):
        h, r, t = list(map(int, line.strip().split('\t')))
        train_data.add((h, r, t))
        trfreq[r] += 1
    train_data = list(train_data)
    for r in trfreq:
        trfreq[r] = args.train_size / (float(trfreq[r]) * len(trfreq))

    # load dev
    tool.trace('load dev')
    dev_data = list()
    for line in open(args.dev_file):
        h, r, t, l = list(map(int, line.strip().split('\t')))
        if h not in glinks or t not in glinks:
            continue
        dev_data.append((h, r, t, l))
    print('dev size:', len(dev_data))

    # load test
    tool.trace('load test')
    test_data = list()
    for line in open(args.test_file):
        h, r, t, l = list(map(int, line.strip().split('\t')))
        if h not in glinks or t not in glinks:
            continue
        test_data.append((h, r, t, l,))
    print('test size:', len(test_data))


def generator_train_with_corruption(args):
    skip_rate = args.train_size / float(len(train_data))

    positive, negative = list(), list()
    random.shuffle(train_data)
    for i in range(len(train_data)):
        h, r, t = train_data[i]
        if (-r, t) in black_set:
            continue
        if (h, r) in black_set:
            continue
        if args.is_balanced_tr:
            if random.random() > trfreq[r]:
                continue
        else:
            if random.random() > skip_rate:
                continue

        # tph/Z
        head_ratio = 0.5
        if args.is_bernoulli_trick:
            head_ratio = tail_per_head[h] / (tail_per_head[h] + head_per_tail[t])
        if random.random() > head_ratio:
            cand = random.choice(candidate_heads[r])
            while cand in gold_heads[(r, t)]:
                cand = random.choice(candidate_heads[r])
            h = cand
        else:
            cand = random.choice(candidate_tails[r])
            while cand in gold_tails[(h, r)]:
                cand = random.choice(candidate_tails[r])
            t = cand
        if len(positive) == 0 or len(positive) <= args.batch_size:
            positive.append(train_data[i])
            negative.append((h, r, t))
        else:
            yield positive, negative
            positive, negative = [train_data[i]], [(h, r, t)]
    if len(positive) != 0:
        yield positive, negative

# ----------------------------------------------------------------------------


def train(args, m, trainer):
    Loss, N = list(), 0
    for positive, negative in generator_train_with_corruption(args):
        trainer.zero_grad()
        loss = m(positive, negative, glinks, gedges)
        Loss.append(float(loss.data)/len(positive))
        N += len(positive)
        loss.backward()
        trainer.step()
        del loss
    return sum(Loss), N


def dump_current_scores_of_devtest(args, m):
    for mode in ['dev', 'test']:
        if mode == 'dev':
            current_data = dev_data
        if mode == 'test':
            current_data = test_data

        scores, accuracy = list(), list()
        for batch in chunked(current_data, args.test_batch_size):
            with torch.no_grad(), torch.set_grad_enabled(False):
                current_score = m.get_scores(batch, glinks, gedges)
            for v, (h, r, t, l) in zip(current_score.data, batch):
                values = (h, r, t, l, v)
                values = map(str, values)
                values = ','.join(values)
                scores.append(values)
                if v < args.threshold:
                    if l == 1:
                        accuracy.append(1.0)
                    else:
                        accuracy.append(0.0)
                else:
                    if l == 1:
                        accuracy.append(0.0)
                    else:
                        accuracy.append(1.0)
            del current_score
        tool.trace('\t ', mode, sum(accuracy) / len(accuracy))
        # if args.margin_file != '':
        #    with open(args.margin_file, 'a') as wf:
        #        wf.write(mode + ':' + ' '.join(scores) + '\n')


def get_sizes(args):
    relation, entity = -1, -1
    for line in open(args.train_file):
        h, r, t = list(map(int, line.strip().split('\t')))
        relation = max(relation, r)
        entity = max(entity, h, t)
    return relation + 1, entity + 1


def main(args):
    initilize_dataset()
    args.rel_size, args.entity_size = get_sizes(args)
    print('relation size:', args.rel_size, 'entity size:', args.entity_size)

    m = get_model(args)
    trainable_parameters = [param for param in m.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=args.l_rate, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epoch_size):
        # scheduler.step()
        trLoss, Ntr = train(args, m, trainer)
        tool.trace('epoch:', epoch, 'tr Loss:', tool.dress(trLoss), Ntr)
        dump_current_scores_of_devtest(args, m)

# ----------------------------------------------------------------------------


def argument():
    p = ArgumentParser()

    # GPU
    p.add_argument('--use_gpu',     '-g',   default=True,   action='store_true')
    p.add_argument('--device',     '-gd',   default=0,       type=int)

    # trian, dev, test, and other filds
    p.add_argument('--train_file',      '-tF',  default='datasets/standard/WordNet11/serialized/train')
    p.add_argument('--dev_file',        '-vF',  default='datasets/standard/WordNet11/serialized/dev')
    p.add_argument('--test_file',       '-eF',  default='datasets/standard/WordNet11/serialized/test')
    p.add_argument('--auxiliary_file',  '-aF',  default='datasets/standard/WordNet11/serialized/train')

    # model parameters (neural network)
    p.add_argument('--nn_model',		'-nn',  default='A')
    p.add_argument('--ent_dim',         '-eD',  default=200,    type=int)
    p.add_argument('--rel_dim',         '-rD',  default=200,    type=int)
    p.add_argument('--ent_size',        '-Es',  default=38194,  type=int)
    p.add_argument('--rel_size',        '-Rs',  default=11,     type=int)

    # loss setting
    p.add_argument('--threshold',            '-tH', default=1200.0,  type=float)

    # lstm
    p.add_argument('--lstm_dim',                     '-lD',     default=200,    type=int)
    p.add_argument('--lstm_layers',                  '-lL',     default=1,      type=int)
    p.add_argument('--lstm_dropout',                 '-lDrop',  default=0,      type=float)
    p.add_argument('--lstm_activate',                '-lA',     default='relu')
    p.add_argument('--is_batch_norm',                '-ib',     default=True)
    p.add_argument('--leaky_relu_negative_slope',    '-ns',     default=0.01,   type=float)
    p.add_argument('--sample_size',                  '-sS',     default=25,     type=int)

    # parameters for negative sampling (corruption)
    p.add_argument('--is_balanced_tr',    '-iBtr',   default=False,   action='store_true')
    # p.add_argument('--is_balanced_dev',   '-nBde',   default=True,   action='store_false')
    p.add_argument('--is_bernoulli_trick', '-iBeT',  default=True,   action='store_false')

    # sizes
    p.add_argument('--train_size',  	'-trS', default=100000,  type=int)
    p.add_argument('--batch_size',		'-bS',  default=5000,    type=int)
    p.add_argument('--test_batch_size', '-tbS', default=20000,   type=int)
    p.add_argument('--epoch_size',      '-eS',  default=1000,    type=int)
    # p.add_argument('--pool_size',		'-pS',  default=128*5,      type=int)

    # optimization
    p.add_argument('--l_rate',        "-lr",  default=0.04,    type=float)
    # p.add_argument('--step_size',     "-ss",  default=30,     type=int)
    # p.add_argument('--gamma',         "-gm",  default=0.8,    type=float)

    # seed to control generaing random variables
    p.add_argument('--seed',        '-seed', default=0,      type=int)

    args = p.parse_args()

    return args


if __name__ == '__main__':

    args = argument()
    print(args)
    print(' '.join(sys.argv))

    if args.seed != -1:
        random.seed()

    main(args)
