import torch

from torch.nn import functional as F, Parameter
import numpy as np
import pickle

from torch.nn.init import xavier_normal_


class DistMult(torch.nn.Module):
    def __init__(self, params):
        super(DistMult, self).__init__()
        self.name = 'Distmult'
        self.emb_e = torch.nn.Embedding(params['num_entities'], params['embedding_dim'], padding_idx=0)
        self.emb_rel = torch.nn.Embedding(params['num_relations'], params['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(params['embedding_dim'])
        self.bn1 = torch.nn.BatchNorm1d(params['embedding_dim'])

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.bn0(self.inp_drop(e1_embedded))
        rel_embedded = self.bn1(self.inp_drop(rel_embedded))

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        self.name = 'Tucker'
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class HypER(torch.nn.Module):
    def __init__(self, kwargs):
        super(HypER, self).__init__()
        self.name = 'Hyper'
        self.in_channels = 1
        self.out_channels = kwargs["conv_out"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(kwargs['num_entities'], kwargs['embedding_dim'], padding_idx=0)
        self.R = torch.nn.Embedding(kwargs['num_relations'], kwargs['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(kwargs['embedding_dim'])
        self.register_parameter('b', Parameter(torch.zeros(kwargs['num_entities'])))
        fc_length = (1 - self.filt_h + 1) * (kwargs['embedding_dim'] - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, kwargs['embedding_dim'])
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(kwargs['embedding_dim'], fc1_length)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))
        r = self.R(r_idx)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


class ConvE(torch.nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        self.name = 'Conve'
        self.emb_e = torch.nn.Embedding(args['num_entities'], args['embedding_dim'], padding_idx=0)
        self.emb_rel = torch.nn.Embedding(args['num_relations'], args['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args['input_dropout'])
        self.hidden_drop = torch.nn.Dropout(args['hidden_dropout'])
        self.feature_map_drop = torch.nn.Dropout2d(args['feature_map_dropout'])
        self.loss = torch.nn.BCELoss()

        self.emb_dim1 = args['embedding_dim'] // 5
        self.emb_dim2 = args['embedding_dim'] // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, args['conv_out'], (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args['conv_out'])
        self.bn2 = torch.nn.BatchNorm1d(args['embedding_dim'])
        self.register_parameter('b', Parameter(torch.zeros(args['num_entities'])))
        self.fc = torch.nn.Linear(args['projection_size'], args['embedding_dim'])

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class Complex(torch.nn.Module):
    def __init__(self, args):
        super(Complex, self).__init__()
        self.name = 'Complex'
        self.num_entities = args['num_entities']
        self.num_relations = args['num_relations']
        self.embedding_dim = args['embedding_dim']

        self.emb_e_real = torch.nn.Embedding(self.num_entities, self.embedding_dim,
                                             padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args['input_dropout'])
        self.loss = torch.nn.BCELoss()

        self.bn0_1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_4 = torch.nn.BatchNorm1d(self.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.bn0_1(self.inp_drop(e1_embedded_real))
        rel_embedded_real = self.bn0_2(self.inp_drop(rel_embedded_real))
        e1_embedded_img = self.bn0_3(self.inp_drop(e1_embedded_img))
        rel_embedded_img = self.bn0_4(self.inp_drop(rel_embedded_img))

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class ConEx(torch.nn.Module):
    """Convolutional Complex Knowledge Graph Embeddings"""

    def __init__(self, params=None):
        super(ConEx, self).__init__()
        self.name = 'Conex'

        self.embedding_dim = params['embedding_dim']
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']
        self.hidden_size = params['projection_size']

        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.hidden_drop = torch.nn.Dropout(params['hidden_dropout'])
        self.feature_map_drop = torch.nn.Dropout2d(params['feature_map_dropout'])

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=params['conv_out'], kernel_size=(4, 4), stride=1,
                                     padding=0, bias=True)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(params['conv_out'])
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn3_1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn3_2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn3_3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn3_4 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.fc = torch.nn.Linear(self.hidden_size, self.embedding_dim)  # Hard compression.

        self.emb_e_real = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1)

        rel_embedded_real = self.emb_rel_real(rel)
        e1_embedded_img = self.emb_e_img(e1)

        rel_embedded_img = self.emb_rel_img(rel)

        x = torch.cat([e1_embedded_real.view(-1, 1, 1, self.embedding_dim),
                       rel_embedded_real.view(-1, 1, 1, self.embedding_dim),
                       e1_embedded_img.view(-1, 1, 1, self.embedding_dim),
                       rel_embedded_img.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        e1_embedded_real = self.bn3_1(self.inp_drop(e1_embedded_real))

        rel_embedded_real = self.bn3_2(self.inp_drop(rel_embedded_real))
        e1_embedded_img = self.bn3_3(self.inp_drop(e1_embedded_img))

        rel_embedded_img = self.bn3_4(self.inp_drop(rel_embedded_img))


        realrealreal = torch.mm(x * e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(x * e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))

        imgrealimg = torch.mm(x * e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(x * e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))

        logits_complEx = realrealreal + realimgimg + imgrealimg - imgimgreal

        pred = torch.sigmoid(logits_complEx)

        return e1_embedded_real, e1_embedded_img, pred


class ConExWithNorm(torch.nn.Module):
    """Convolutional Complex Knowledge Graph Embeddings"""

    def __init__(self, params):
        super(ConExWithNorm, self).__init__()

        self.embedding_dim = params['embedding_dim']
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']

        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.hidden_drop = torch.nn.Dropout(params['hidden_dropout'])
        self.feature_map_drop = torch.nn.Dropout2d(params['feature_map_dropout'])

        self.register_parameter('b', Parameter(torch.zeros(self.num_entities)))
        self.register_parameter('b_image', Parameter(torch.zeros(self.num_entities)))

        self.conv1 = torch.nn.Conv2d(1, params['conv_out'], (4, 4), 1, 0, bias=True)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(params['conv_out'])
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.hidden_size = params['projection_size']
        self.fc = torch.nn.Linear(self.hidden_size, self.embedding_dim)

        self.emb_e_real = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1)
        rel_embedded_real = self.emb_rel_real(rel)
        e1_embedded_img = self.emb_e_img(e1)
        rel_embedded_img = self.emb_rel_img(rel)

        x = torch.cat([e1_embedded_real.view(-1, 1, 1, self.embedding_dim),
                       rel_embedded_real.view(-1, 1, 1, self.embedding_dim),
                       e1_embedded_img.view(-1, 1, 1, self.embedding_dim),
                       rel_embedded_img.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        realrealreal = torch.mm(x * e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(x * e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))

        imgrealimg = torch.mm(x * e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(x * e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))

        logits_complEx = realrealreal + realimgimg + imgrealimg - imgimgreal

        pred = torch.sigmoid(logits_complEx)

        return pred
