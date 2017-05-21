#!/usr/bin/env python
import chainer
import chainer.functions as F
import chainer.links as L
import nets

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, _train=True, _dr=0.5):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_units),  # n_units -> n_units
            l4=L.Linear(None, n_units),  # n_units -> n_units
            l5=L.Linear(None, n_out),  # n_units -> n_out
        )
        self.train = _train
        self.dr = _dr

    def set_train_state(self, _train):
        self.train = _train
    def set_dropout_r(self, _dr):
        self.dr = _dr

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)), train=self.train, ratio=self.dr)
        h2 = F.dropout(F.relu(self.l2(h1)), train=self.train, ratio=self.dr)
        h3 = F.dropout(F.relu(self.l3(h2)), train=self.train, ratio=self.dr)
        h4 = F.dropout(F.relu(self.l4(h3)), train=self.train, ratio=self.dr)
        return self.l5(h4)

