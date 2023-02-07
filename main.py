from core import Trainer, Certifier 

lambd = 3.5*3**0.5
noise = 'split'
dataset = 'cifar'

trn = Trainer(dataset, noise, lambd)
trn.train()
c = Certifier(lambd, noise)
ret = c.certify(trn.model, dataset,skip=20, bs=1024, n=10**4)

# ret is a list of triples (y, pred, r)
#
# y   : actual label
# pred: predicted label
# r   : certified radius for pred.

