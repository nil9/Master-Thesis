from collections import defaultdict
import numpy as np
import torch
from models import ConEx, ConExWithNorm
from helper_classes import Data
from tqdm import tqdm
import pickle

kg_path = 'KGs/UMLS'
data_dir = "%s/" % kg_path
model_path = 'PretrainedModels/UMLS/conex_umls.pt'

d = Data(data_dir=data_dir, reverse=True)


class Reproduce:

    def __init__(self):
        self.cuda = False

        self.batch_size = 128

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        print("the data index are:",data_idxs)

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
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        print("self.entity_idxs from evaluate",self.entity_idxs )
        hits = []
        ranks = []
        rank_per_relation = dict()

        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        print("Length of test Data index is:",len(test_data_idxs))

        inverse_relation_idx = dict(zip(self.relation_idxs.values(), self.relation_idxs.keys()))

        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            final_embedding = {}
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            e1_embedded_real, e1_embedded_img, predictions = model.forward(e1_idx, r_idx)

            ## convert the tensors to list
            e1_idx_list = e1_idx.tolist()
            ## replace the index with original item identifier
            for i in range(0, len(e1_idx_list)):
                for k, v in self.entity_idxs.items():
                    if e1_idx_list[i] == v:
                        e1_idx_list[i] = k
            print("Length of e1_idx is:",len(e1_idx_list))
            e1_embedded_real_list = e1_embedded_real.tolist()
            e1_embedded_img_list = e1_embedded_img.tolist()
            ## store the embedding with item identifier key in a dictionary
            for i in range(0, len(e1_idx_list)):
                entity_dict = {}
                cocat_emb = e1_embedded_real_list[i] + e1_embedded_img_list[i]
                entity_dict.update({e1_idx_list[i]: cocat_emb})
                with open('item_index.p', 'ab') as f:
                    pickle.dump(entity_dict, f)
                del cocat_emb

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

        #print('Hits @10: {0}'.format(np.mean(hits[9])))
        #print('Hits @3: {0}'.format(np.mean(hits[2])))
        #print('Hits @1: {0}'.format(np.mean(hits[0])))
        #print('Mean rank: {0}'.format(np.mean(ranks)))
        #print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        print('##########################################################\n')
        if top_10_per_rel:

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

                if '_reverse' in rel:
                    continue


                print('{0}. {1} => freq {2}'.format(top, rel, v))

                selected_rels.append(rel)
                if top == 200:
                    break
                top += 1

            for rel in selected_rels:

                if '_reverse' in rel:
                    continue

                # data conversion
                rank_per_relation[rel] = np.array(rank_per_relation[rel])

                print('{0}: Mean Reciprocal Rank: {1}'.format(rel, np.mean(1. / rank_per_relation[rel])))

                hit10 = (rank_per_relation[rel] <= 10).sum() / len(rank_per_relation[rel])
                hit3 = (rank_per_relation[rel] <= 3).sum() / len(rank_per_relation[rel])
                hit1 = (rank_per_relation[rel] == 1).sum() / len(rank_per_relation[rel])

                print('H@10 for {0}: {1}'.format(rel, hit10))
                print('H@3 for {0}: {1}'.format(rel, hit3))
                print('H@1 for {0}: {1}'.format(rel, hit1))

        else:

            for relations, ranks_ in rank_per_relation.items():


                if '_reverse' in relations:
                    continue

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
        #print("The entities",d.entities)
        #print(self.entity_idxs )
        ##################       Dumping the entity index mapping ###########
        #print("Length of item-index mapping is:",self.entity_idxs)
        #with open('relation_map.p', 'ab') as f:
           # pickle.dump(self.entity_idxs, f)

        params = {'num_entities': len(self.entity_idxs),
                  'num_relations': len(self.relation_idxs),
                  'embedding_dim': 100,
                  'input_dropout': 0.4,
                  'conv_out': 8,
                  'hidden_dropout': 0.4,
                  'projection_size': 776,
                  'feature_map_dropout': 0.4}

        model = ConEx(params)

        #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

        if self.cuda:
            model.cuda()
        print('Number of free parameters: ', sum([p.numel() for p in model.parameters()]))

        print('Test Results')
        self.evaluate(model, d.test_data,True)


Reproduce().reproduce()

