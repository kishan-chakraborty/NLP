from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

with open(r'toxic_vect.pkl', 'rb') as f:
    tox= pickle.load(f)

with open(r'severe_toxic_vect.pkl', 'rb') as f:
    sev= pickle.load(f)

with open(r'obscene_vect.pkl', 'rb') as f:
    obs= pickle.load(f)

with open(r'insult_vect.pkl', 'rb') as f:
    ins= pickle.load(f)

with open(r'threat_vect.pkl', 'rb') as f:
    thr= pickle.load(f)

with open(r'identity_hate_vect.pkl', 'rb') as f:
    idh= pickle.load(f)

with open(r'toxic_model.pkl', 'rb') as f:
    tox_model= pickle.load(f)

with open(r'severe_toxic_model.pkl', 'rb') as f:
    sev_model= pickle.load(f)

with open(r'obscene_model.pkl', 'rb') as f:
    obs_model= pickle.load(f)

with open(r'insult_model.pkl', 'rb') as f:
    ins_model= pickle.load(f)

with open(r'threat_model.pkl', 'rb') as f:
    thr_model= pickle.load(f)

with open(r'identity_hate_model.pkl', 'rb') as f:
    idh_model= pickle.load(f)

models= [tox_model, sev_model, obs_model, ins_model, thr_model, idh_model]
vectors= [tox, sev, obs, ins, thr, idh]
labels= ['Toxicity', 'Severe toxicity', 'Obscenity', 'Insult level', 'Threat level', 'Identity hate level']


def predict(statement, vect, model, label):
    vect_stm= vect.transform([statement])
    print(f'{label}:\t\t {model._predict_proba_lr(vect_stm)[0][1]*100}%')

statement= input('Enter your comment\n')
for i in range(len(labels)):
    predict(statement, vectors[i], models[i], labels[i])