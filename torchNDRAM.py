import torch
print(('not using gpu', 'using gpu')[torch.cuda.is_available()])
from tqdm import tqdm
import numpy as np
import time
import pickle

## proposed in NDRAM paper (Chartier, S.; 2005), kept constant
h = 197e-5
delta = 0.4

## ndram activation as included in the paper
def ndram_activation(x):
    torch.where(x < -1, -1, x)
    torch.where(x > 1, 1, x)
    return (1 + delta) * x - delta * x**3

## cosine similarity for error
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

## get an empty weight matrix
def initial_weights(n: int):
    return torch.zeros([n,n])

## transmit stimuli n times, according to activation and weight matrix
def transmission_n(W, x0, n):
    xt = x0
    for i in range(n):
        xt = ndram_activation(torch.mv(W, xt))
    return xt

## update weight matrix
def learn(W, x0_op, xt):
    W += h * (x0_op - torch.outer(xt, xt))
    return W

## transmit stimuli and update weight matrix
def transmit_and_learn(W, x0, x1, n, x0_op):
    xt = transmission_n(W, x0, n)
    W = learn(W, x0_op, xt)
    return cos_sim(x1, xt)

#########    

class AutoNDRAMTrain:
    def __init__(self):
        self.W = None
        self.tf = None

    def fit(self, stimuli_in, cos_thresh=0.99, transmit_factor=1):
        stimuli = [torch.Tensor(s) for s in stimuli_in]
        n_stimuli = len(stimuli)
        x0_ops = [torch.outer(s, s) for s in stimuli]
        self.W = initial_weights(len(stimuli_in[0]))
        self.tf = transmit_factor

        avg_cos = 0
        count = 0
        elapsed = 0
        while(avg_cos < cos_thresh or np.isnan(avg_cos)):
            cos = 0
            c_desc = "{fcount}:, cosine similarity: {favg_cos:.4f}, elapsed: {felapsed:.2f} seconds".format(fcount=count, favg_cos=avg_cos, felapsed=elapsed)
            for i in tqdm(range(n_stimuli), desc=c_desc):
                start = time.time()
                cos += transmit_and_learn(self.W, stimuli[i], stimuli[i], self.tf, x0_ops[i])
                elapsed += (time.time() - start)
            count += 1
            avg_cos = cos / n_stimuli
        # self.W = torch.nn.functional.normalize(self.W, dim = 1)
    
    # # # ## get nearest weight by cosine similarity
    # # # def nearest_weight(self, stimulus):
    # # #     return self.nearest_weights(stimulus, 1)[0]

    # # # ## get nearest n weight by cosine similarity
    # # # def nearest_weights(self, stimulus, n):
    # # #     res = self.predict(torch.Tensor(stimulus))
    # # #     dist_list = [[i, r] for i, r in enumerate(res)]
    # # #     dist_list = sorted(dist_list, key=lambda x: x[1], reverse=True)[:n]
    # # #     return [i[0] for i in dist_list]

    # # # ## get weights above random threshold (1 / n_labels), by cosine similarity
    # # # def nearest_weights_thresh(self, stimulus, n_labels):
    # # #     thresh = 1 / 768
    # # #     res = self.predict(torch.Tensor(stimulus))
    # # #     dist_list = [[i, r] for i, r in enumerate(res) if r > thresh]
    # # #     dist_list = sorted(dist_list, key=lambda x: x[1], reverse=True)
    # # #     return [i[0] for i in dist_list]

    def save(self, filepath):
        filehandler = open(filepath, 'wb')
        pickle.dump(self.__dict__, filehandler, 2)
        filehandler.close()

    def load(self, filepath):
        f = open(filepath, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict)
    
class AutoNDRAMTest:
    def __init__(self):
        self.W = None
        self.tf = None

    def load(self, filepath):
        f = open(filepath, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict)

    ## activate and transmit of test stimuli for prediction
    def predict(self, stimulus):
        y = transmission_n(self.W, torch.Tensor(stimulus), self.tf)
        return y, int(torch.argmax(y))
    
class AutoNDRAM:
    def __init__(self):
        self.train = AutoNDRAMTrain()
        self.test = AutoNDRAMTest()

        self.W = None
        self.tf = None

    def fit(self, stimuli_in, cos_thresh=0.99, transmit_factor=1):
        self.train.fit(stimuli_in=stimuli_in, cos_thresh=cos_thresh, transmit_factor=transmit_factor)
        self.W = self.train.W
        self.tf = self.train.tf
        self.test.W = self.train.W
        self.test.tf = self.train.tf

    def save(self, filepath):
        self.train.save(filepath=filepath)

    def load(self, filepath):
        self.train.load(filepath=filepath)
        self.test.load(filepath=filepath)

        self.W = self.test.W
        self.tf = self.test.tf

    def predict(self, stimulus):
        return self.test.predict(stimulus=stimulus)