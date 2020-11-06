import json

from helper_classes import *
from helper_funcs import *

from collections import defaultdict
from models import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse


class Experiment:

    def __init__(self, *, model, learning_rate=0.0005, embedding_dim,
                 num_iterations, batch_size=128, decay_rate=0., conv_out=2, projection_size=10,
                 input_dropout=0.4, feature_map_dropout=0.4,
                 hidden_dropout=0.4, label_smoothing=0.1, cuda=True):

        self.model = model
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = torch.cuda.is_available()
        self.entity_idxs, self.relation_idxs, self.scheduler = None, None, None

        # Params stored in kwargs for creating pretrained models with unique names.
        self.kwargs = {'embedding_dim': embedding_dim,
                       'learning_rate': learning_rate,
                       'batch_size': batch_size,
                       'conv_out': conv_out,
                       'input_dropout': input_dropout,
                       'hidden_dropout': hidden_dropout,
                       'projection_size': projection_size,
                       'feature_map_dropout': feature_map_dropout,
                       'label_smoothing': label_smoothing,
                       'decay_rate': decay_rate}

        self.storage_path, _ = create_experiment_folder()

        self.logger = create_logger(name=self.model, p=self.storage_path)

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

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

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

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        self.logger.info('Hits @10: {0}'.format(np.mean(hits[9])))

        self.logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        self.logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        # print('Mean rank: {0}'.format(np.mean(ranks)))
        self.logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self, d_info):

        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)

        self.kwargs.update({'num_entities': len(self.entity_idxs),
                            'num_relations': len(self.relation_idxs)})
        self.kwargs.update(d_info)

        self.logger.info("Info pertaining to dataset:{0}".format(d_info['dataset']))
        self.logger.info("Number of triples in training data:{0}".format(len(d.train_data)))
        self.logger.info("Number of triples in validation data:{0}".format(len(d.valid_data)))
        self.logger.info("Number of triples in testing data:{0}".format(len(d.test_data)))
        self.logger.info("Number of entities:{0}".format(len(self.entity_idxs)))
        self.logger.info("Number of relations:{0}".format(len(self.relation_idxs)))

        self.logger.info("HyperParameter Settings:{0}".format(self.kwargs))

        model = None
        if self.model == 'Conex':
            model = ConEx(self.kwargs)
        elif self.model == 'Distmult':
            model = DistMult(self.kwargs)
        elif self.model == 'Complex':
            model = Complex(self.kwargs)
        elif self.model == 'Conve':
            model = ConvE(self.kwargs)
        elif self.model == 'Tucker':
            model = TuckER(self.kwargs)
        elif self.model == 'HypER':
            model = HypER(self.kwargs)
        else:
            raise ValueError

        self.logger.info(model)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        self.logger.info("{0} starts training".format(model.name))
        num_param = sum([p.numel() for p in model.parameters()])
        self.logger.info("'Number of free parameters: {0}".format(num_param))

        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                self.scheduler.step()

            if it % 500 == 0:
                self.logger.info('Iteration:{0} with Average loss{1}'.format(it, np.mean(losses)))
                model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
                with torch.no_grad():  # Important:
                    self.logger.info("Validation:")
                    self.evaluate(model, d.valid_data)

        with open(self.storage_path + '/settings.json', 'w') as file_descriptor:
            json.dump(self.kwargs, file_descriptor)

        self.logger.info("Testing:")
        self.evaluate(model, d.test_data)
        torch.save(model.state_dict(), self.storage_path + '/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Conex', nargs="?",
                        help="Models:Conex, Distmult, Complex, Tucker, Conve or HypER")
    parser.add_argument("--dataset", type=str, default="KGs/KINSHIP", nargs="?",
                        help="Which dataset to use with KGs/XXX: WN18RR, FB15k-237, UMLS or KINSHIP.")
    parser.add_argument("--embedding_dim", type=int, default=20, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_iterations", type=int, default=1000, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--conv_out", type=int, default=32, nargs="?",
                        help="Number of channels for 2D-Convolution.")
    parser.add_argument("--projection_size", type=int, default=576, nargs="?",
                        help="Projection size after feature map.")
    parser.add_argument("--input_dropout", type=float, default=0.1, nargs="?",
                        help="Dropout rate before 2D-Convolution.")
    parser.add_argument("--feature_map_dropout", type=float, default=0.1, nargs="?",
                        help="Dropout rate after 2D-Convolution.")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, nargs="?",
                        help="Dropout rate before projection.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--reverse", type=bool, default=False, nargs="?",
                        help="Data Augmentation by generating reciprocal relations")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    d = Data(data_dir=data_dir, reverse=args.reverse)

    experiment = Experiment(model=args.model, num_iterations=args.num_iterations, batch_size=args.batch_size,
                            embedding_dim=args.embedding_dim,
                            learning_rate=args.lr, decay_rate=args.dr, conv_out=args.conv_out,
                            projection_size=args.projection_size,
                            input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
                            feature_map_dropout=args.feature_map_dropout, label_smoothing=args.label_smoothing)
    experiment.train_and_eval(d.info)
