import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

# c_p_e = np.load('../domain_clf_data/cooking_phrase_embeddings.npy')
c_u_e = np.load('../domain_clf_data/cooking_utterance_embeddings.npy')
# c_p_e = np.load('../domain_clf_data/distilroberta_cooking_phrase_embeddings.npy')
# c_u_e = np.load('../domain_clf_data/distilroberta_cooking_utterance_embeddings.npy')

# d_p_e = np.load('../domain_clf_data/diy_phrase_embeddings.npy')
d_u_e = np.load('../domain_clf_data/diy_utterance_embeddings.npy')
# d_p_e = np.load('../domain_clf_data/distilroberta_diy_phrase_embeddings.npy')
# d_u_e = np.load('../domain_clf_data/distilroberta_diy_utterance_embeddings.npy')

# h_p_e = np.load('../domain_clf_data/hobby_phrase_embeddings.npy')
h_u_e = np.load('../domain_clf_data/hobby_utterance_embeddings.npy')
# h_p_e = np.load('../domain_clf_data/distilroberta_hobby_phrase_embeddings.npy')
# h_u_e = np.load('../domain_clf_data/distilroberta_hobby_utterance_embeddings.npy')

# f_p_e = np.load('../domain_clf_data/financial_phrase_embeddings.npy')
# f_u_e = np.load('../domain_clf_data/financial_utterance_embeddings.npy')
# f_p_e = np.load('../domain_clf_data/distilroberta_financial_phrase_embeddings.npy')
# f_u_e = np.load('../domain_clf_data/distilroberta_financial_utterance_embeddings.npy')

# m_p_e = np.load('../domain_clf_data/medical_phrase_embeddings.npy')
# m_u_e = np.load('../domain_clf_data/medical_utterance_embeddings.npy')
# m_p_e = np.load('../domain_clf_data/distilroberta_medical_phrase_embeddings.npy')
# m_u_e = np.load('../domain_clf_data/distilroberta_medical_utterance_embeddings.npy')

# l_p_e = np.load('../domain_clf_data/legal_phrase_embeddings.npy')
# l_u_e = np.load('../domain_clf_data/legal_utterance_embeddings.npy')
# l_p_e = np.load('../domain_clf_data/distilroberta_legal_phrase_embeddings.npy')
# l_u_e = np.load('../domain_clf_data/distilroberta_legal_utterance_embeddings.npy')

# dan_p_e = np.load('../domain_clf_data/danger_phrase_embeddings.npy')
# dan_u_e = np.load('../domain_clf_data/danger_utterance_embeddings.npy')
# dan_p_e = np.load('../domain_clf_data/distilroberta_danger_phrase_embeddings.npy')
# dan_u_e = np.load('../domain_clf_data/distilroberta_danger_utterance_embeddings.npy')

# cooking_phrases = pd.read_csv('../domain_clf_data/cooking_phrases.txt')
# cooking_phrases = cooking_phrases[cooking_phrases.columns[0]].to_list()
# cooking_phrases = list(set(cooking_phrases))
# c_e = model.encode(cooking_phrases)

# diy_phrases = pd.read_csv('../domain_clf_data/diy_phrases.txt')
# diy_phrases = diy_phrases[diy_phrases.columns[0]].to_list()
# diy_phrases = list(set(diy_phrases))
# d_e = model.encode(diy_phrases)

# hobby_phrases = pd.read_csv('../domain_clf_data/hobby_phrases.txt')
# hobby_phrases = hobby_phrases[hobby_phrases.columns[0]].to_list()
# hobby_phrases = list(set(hobby_phrases))
# h_e = model.encode(hobby_phrases)

# from torchNDRAM import AutoNDRAM

# o_data = [c_p_e, d_p_e, h_p_e, f_p_e, m_p_e, l_p_e, dan_p_e]
# X_train = np.vstack(o_data)
# y_train = ['cooking' for i in c_p_e] + ['diy' for i in d_p_e] + ['hobby' for i in h_p_e] + ['financial' for i in f_p_e] + ['medical' for i in m_p_e] + ['legal' for i in l_p_e] + ['danger' for i in dan_p_e]

# f_data = [c_u_e, d_u_e, h_u_e, f_u_e, m_u_e, l_u_e, dan_u_e]
# X_test = np.vstack(f_data)
# y_test = ['cooking' for i in c_u_e] + ['diy' for i in d_u_e] + ['hobby' for i in h_u_e] + ['financial' for i in f_u_e] + ['medical' for i in m_u_e] + ['legal' for i in l_u_e] + ['danger' for i in dan_u_e]