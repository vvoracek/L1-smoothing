import torch
import torchvision 
from collections import defaultdict, Counter
import math 
from tqdm import tqdm 
from statsmodels.stats.proportion import proportion_confint
from datasets import get_dataset
import models 



class UniformNoise():
    def __init__(self, lambd):
        self.lambd = lambd 
    
    def get_noise_batch(self,x):
        device = x.get_device()
        return (torch.rand(x.shape, device=device)-0.5)*self.lambd*2 + x

class SplittingNoise():
    def __init__(self, lambd):
        self.lambd = lambd 
    
    def get_noise_batch(self, x):
        device = x.get_device()
        if(self.lambd >= 0.5):
            s = torch.rand(x.shape, device=device)*2*self.lambd
            ind = (x > s).float()
            s = torch.clamp(s, max=1)
            return (s + ind)/2
        else:
            s = torch.rand(x.shape, device=self.device)*2*self.lambd
            ceils = torch.ceil((x-s)/(2.*self.lambd))*2.*self.lambd + s
            upper = torch.clamp(ceils, max=1)
            lower = torch.clamp(ceils - 2.*self.lambd, min=0)
            return (lower+upper)/2

class Certifier():
    def __init__(self, lambd, noise):
        if(noise == 'split'):
            self.noise = SplittingNoise(lambd)
        elif(noise == 'uniform'):
            self.noise = UniformNoise(lambd)
        else:
            raise ValueError("noise should be split or uniform, received " + str(noise))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_noise_batch = self.noise.get_noise_batch

    def certify(self, model, dataset, bs = 32, n0=128, n=100000, alpha=0.001, skip=20):
        model.eval()
        n0 = math.ceil(n0/bs)*bs 
        n = math.ceil(n/bs)*bs 
        dataset = get_dataset(dataset,'test') 
        ret = []

        with torch.no_grad():
            for idx in tqdm(range(0,len(dataset),skip)):
                counts = Counter(), Counter()
                X, y = dataset[idx]
                X = X.to(self.device)
                X = X.repeat((bs, 1, 1, 1))
                for bidx in range((n+n0) // bs):
                    x = self.get_noise_batch(X)
                    preds = model(x).argmax(-1)
                    counts[bidx >= n0//bs].update(preds.detach().tolist())

                pred = counts[0].most_common(1)[0][0]
                top2 = counts[1].most_common(2)
                ca = top2[0][1]
                if(len(top2) > 1):
                    cb = top2[1][1]
                else:
                    cb = 0

                diff = (proportion_confint(ca, n, alpha = alpha, method='beta')[0] -
                        proportion_confint(cb, n, alpha = alpha, method='beta')[1])
                X = torch.maximum(X[0], 1-X[0]).reshape(-1)
                ret.append((y, pred, self.radius(diff, X.detach().tolist())))
            return ret 

    def radius(self, diff, x):
        x = sorted(x, reverse=True)
        lambd = self.noise.lambd
        t = 1-diff/2

        c = 0
        p = 1
        sm = 0
        while(1):
            u = 2*lambd*(1-t/p)
            if(u <= x[c]):
                return u + sm
            sm += x[c]
            p *= (1-x[c]/2/lambd)
            c += 1


class Trainer():
    def __init__(self, dataset, noise, lambd):

        if(noise == 'split'):
            self.noise = SplittingNoise(lambd)
        elif(noise == 'uniform'):
            self.noise = UniformNoise(lambd)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_noise_batch = self.noise.get_noise_batch
        self.dataset = dataset 

        if(dataset == 'cifar'):
            self.model = models.WideResNet(dataset, self.device)
        elif(dataset == 'imagenet'):
            self.model = models.ResNet(dataset, self.device)

        self.model.train()

    def train(self, bs = 1000, lr = 0.1, num_epochs = 120, stability = False):
        train_loader = torch.utils.data.DataLoader(get_dataset(self.dataset, "train"),
                                shuffle=True,
                                batch_size=bs,
                                num_workers=2,
                                pin_memory=False)

        optimizer = torch.optim.SGD(self.model.parameters(),
                            lr=lr,
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)

        annealer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        acc = 0
        for epoch in tqdm(range(0,num_epochs)):
            for idx, (x,y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                if(not stability):
                    loss = self.model.loss(self.get_noise_batch(x),y).mean()
                else:
                    pred1 = self.model.forecast(self.model.forward(self.get_noise_batch(x)))
                    pred2 = self.model.forecast(self.model.forward(self.get_noise_batch(x)))
                    loss = -pred1.log_prob(y) -pred2.log_prob(y)+ 12.0 * torch.distributions.kl_divergence(pred1, pred2)
                    loss = loss.mean()
        
                acc += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(acc)
            acc = 0
            annealer.step()
 
