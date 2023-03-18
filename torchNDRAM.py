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

# # # # # # ## DEMO:

# ## arbitrary stimuli
# stimuli_out = [
#     [1,0,1,0],
#     [0,1,0,1],
#     [0,1,1,0],
#     [1,0,0,1],
#     # [-1,0,-1,1],
#     # [0,-1,-1,0]
# ]

# stimuli_in = [
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,0],
#     [0,0,0,1],
#     # [-1,0,-1,1],
#     # [0,-1,-1,0]
# ]

# otto = cosNDRAM(stimuli_in, stimuli_in)
# for s in stimuli_out:
#     print(otto.predict(s))
# # # # c_e = np.load('sbert_cooking_embeddings_022323.npy')
# # # # d_e = np.load('sbert_diy_embeddings_022323.npy')
# # # # h_e = np.load('sbert_hobby_embeddings_022323.npy')
# # # # gpt_e = np.load('gpt_utterance_sbert_embeddings.npy')
# # # # o_data = [c_e, d_e, h_e]
# # # # otto = AutoNDRAM(gpt_e, mse_thresh=0.001)
# # # # n = 256
# # # # a = sorted(otto.nearest_weights(c_e[0], n))
# # # # b = sorted(otto.nearest_weights(d_e[0], n))
# # # # c = sorted(otto.nearest_weights(h_e[0], n))
# # # # listo = [a,b,c]
# # # # # print(listo)
# # # # # print(listo[0])

# # # # delete_list = []
# # # # for i in range(768):
# # # #     sum = 0
# # # #     for l in listo:
# # # #         sum += i in l
# # # #     if sum > 1:
# # # #         delete_list.append(i)

# # # # new_listo = []
# # # # for l in listo:
# # # #     tmp = []
# # # #     for j in l:
# # # #         if j not in delete_list:
# # # #             tmp.append(j)
# # # #     # for d in delete_list:
# # # #     #     if d not in l:
# # # #     #         tmp.append(d)
# # # #     new_listo.append(tmp)
# # # # for l in new_listo:
# # # #     print(len(l), l)
# # # # print(torch.linalg.eig(otto.W)[0])
# # # # # for o in o_data:
# # # # #     for l in new_listo:
# # # # #         print(set([otto.nearest_weight(e) for e in o]) & set(l))

# # # # # for w in otto.W:
# # # # #     print(max(w), min(w))

# # # # # print(sorted(otto.W[0])[:100])
# # # # # print(sorted(otto.W[0])[-100:])
# # # # # print(max(otto.W[1]), np.median(otto.W[1]), min(otto.W[1]))
# # # # # print(set([otto.nearest_weight(e) for e in d_e]) & set(new_listo[1]))
# # # # # print(set([otto.nearest_weight(e) for e in h_e]) & set(new_listo[2]))
# # # # # for e in c_e:
# # # # #     print(otto.nearest_weight(e))
# # # # # for e in d_e:
# # # # #     print(otto.nearest_weight(e))
# # # # # for e in h_e:
# # # # #     print(otto.nearest_weight(e))
# # # # # print(len(delete_list))    
# # # # # for i in range(len(a)):
# # # # # print(len(a), len(b), len(c))
# # # # # print(a, b, c)

# # # # # ## start with a matrix of zeroes (dimensions must match number of features in stimuli)
# # # # # W = initial_weights(len(stimuli[0]))

# # # # # ## convert stimuli to torch tensors
# # # # # inputs = [torch.Tensor(s) for s in stimuli]

# # # # # ## start with high mse value for loop
# # # # # mse = 100

# # # # # while mse > 0.0001:
# # # # #     ## for each stimuli... 
# # # # #     for i, xi in tqdm(enumerate(inputs), desc="mse: %f" % mse):

# # # # #         ## transmit n times, then update weight matrix and mse (readout included in tqdm progress bar)
# # # # #         mse = transmit_and_learn(W, xi, xi, 3)


# # # # # ## sanity check:
# # # # # # s[0] and s[2] are positively associated by stimuli 4 (fifth stimulus) and negatively associated with s[3]
# # # # # # s[0] has no other direct associations, whereas s[2] is positively associated with s[1] by stimulus 5 (sixth stimulus)
# # # # # # 
# # # # # # -> so we expect to see s[1] slim and positive and s[3] slim and negative

# # # # # s = torch.Tensor([1,0,1,0])
# # # # # s /= torch.sum(s)
# # # # # print(torch.mv(W, s))

# # # # # cos_thresh = 0.99
# # # # # from helper import c_u_e, c_p_e, f_data, o_data
# # # # # hebbians = [Hebbian() for i in o_data]
# # # # # ndrams = [AutoNDRAM() for i in o_data]
# # # # # for i, ndram in enumerate(ndrams):
# # # # #     ndram.fit(o_data[i], cos_thresh=cos_thresh)
# # # # #     ndram.save(str(i)+'_' + str(cos_thresh) + '_.model')

# # # # # # clf = AutoNDRAM()
# # # # # # clf.fit(c_p_e, cos_thresh=0.99)
# # # # # # clf.fit(c_p_e, cos_thresh=0.99)
# # # # # categories = ['cooking', 'diy', 'hobby', 'financial', 'medical', 'legal', 'danger']

# # # # # res = [categories[np.argmax([sum(ndram.predict(e)[0]) for ndram in ndrams])] for e in c_u_e]
# # # # # # for e in c_u_e:
# # # # # #     tmp = [sum(ndram.predict(e)[0]) for ndram in ndrams]
# # # # # #     res.append(tmp)

# # # # # print(res)
# # # # #     # for i, domain in enumerate(f_data):
# # # # #     #     print(categories[i])
# # # # #     #     for e in domain:
# # # # #     #         print(sum(ndram.predict(e)[0]))
# # # # #     #     print('\n\n')

# from sklearn.decomposition import PCA
# from helper import c_p_e, c_u_e

# print(len(c_p_e[0]))
# print(len(c_u_e[0]))
# pca = PCA(n_components=768)
# pca.fit(c_p_e)
# print(len())
# print(sum(pca.transform(c_u_e[0])))


# import faiss
# from helper import X_train, c_u_e, d_u_e, h_u_e, f_data

# nlist = 7
# # k = 4
# d = 768
# quantizer = faiss.IndexFlatIP(d)  # the other index
# index = faiss.IndexIVFFlat(quantizer, d, nlist)
# index.train(X_train)
# index.add(X_train)                  # add may be a bit slower as well
# # D, I = index.search(xq, k)     # actual search
# # print(I[-5:])                  # neighbors of the 5 last queries
# # index.nprobe = 10              # default nprobe is 1, try a few more
# # D, I = index.search(xq, k)
# # print(I[-5:])
# centroids = index.quantizer.reconstruct_n(0, index.nlist)
# W = np.vstack(centroids)
# # for domain in f_data:
# #     for e in 
# for e in c_u_e:
#     print(np.argmax(W.dot(e)))
# print('\n')
# for e in d_u_e:
#     print(np.argmax(W.dot(e)))
# print('\n')
# for e in h_u_e:
#     print(np.argmax(W.dot(e)))
