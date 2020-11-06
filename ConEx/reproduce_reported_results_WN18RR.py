from collections import defaultdict
import numpy as np
import torch
from models import ConEx, ConExWithNorm
from helper_classes import Data

kg_path = 'KGs/WN18RR'
data_dir = "%s/" % kg_path
model_path = 'PretrainedModels/WN18RR/conex_WN18RR.pt'

d = Data(data_dir=data_dir, reverse=False)


class Reproduce:

    def __init__(self):
        self.cuda = False

        self.batch_size = 128

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data, top_10_per_rel=True):
        hits = []
        ranks = []
        rank_per_relation = dict()
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)

        inverse_relation_idx = dict(zip(self.relation_idxs.values(), self.relation_idxs.keys()))

        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                rank_per_relation.setdefault(inverse_relation_idx[data_batch[j][1]], []).append(rank + 1)

                for hits_level in range(10):
                    val = 0.0
                    if rank <= hits_level:
                        val = 1.0
                    hits[hits_level].append(val)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        print('##########################################################\n')

        # Get frequencies
        freq = dict()
        for i in data:
            rel = i[1]
            if rel in freq:
                freq[rel] += 1
            else:
                freq[rel] = 1
        freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

        selected_rels = []

        # Select top 10
        top = 1
        for rel, v in freq.items():
            print('{0}. {1} => freq {2}'.format(top, rel, v))

            selected_rels.append(rel)
            if top == 20:
                break
            top += 1

        for relations, ranks_ in rank_per_relation.items():

            rank_per_relation[relations] = np.array(ranks_)

            print('{0}: Mean Reciprocal Rank: {1}'.format(relations, np.mean(1. / rank_per_relation[relations])))

            hit10 = (rank_per_relation[relations] <= 10).sum() / len(rank_per_relation[relations])
            hit3 = (rank_per_relation[relations] <= 3).sum() / len(rank_per_relation[relations])
            hit1 = (rank_per_relation[relations] == 1).sum() / len(rank_per_relation[relations])

            print('H@10 for {0}: {1}'.format(relations, hit10))
            print('H@3 for {0}: {1}'.format(relations, hit3))
            print('H@1 for {0}: {1}'.format(relations, hit1))




    def reproduce(self, ):

        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        params = {'num_entities': len(self.entity_idxs),
                  'num_relations': len(self.relation_idxs),
                  'embedding_dim': 100,
                  'input_dropout': 0.1,
                  'conv_out': 64,
                  'hidden_dropout': 0.2,
                  'projection_size': 6208,
                  'feature_map_dropout': 0.1}

        model = ConEx(params)

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

        if self.cuda:
            model.cuda()
        print('Number of free parameters: ', sum([p.numel() for p in model.parameters()]))

        print('Test Results')
        self.evaluate(model, d.test_data,False)


Reproduce().reproduce()
