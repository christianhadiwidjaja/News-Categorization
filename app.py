import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
import numpy as np

from tkinter import *
import tkinter.scrolledtext as ScrolledText
from scipy.spatial import distance_matrix

k = 1
model2 = SentenceTransformer('stsb-roberta-base')

## for data
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
import re
from sklearn import model_selection,metrics
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K## for bert language model
import tensorflow
import transformers
import shutil

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

## inputs
idx = layers.Input((50), dtype="int32", name="input_idx")
masks = layers.Input((50), dtype="int32", name="input_masks")
segments = layers.Input((50), dtype="int32", name="input_segments")## pre-trained bert
nlp = transformers.TFBertModel.from_pretrained("bert-base-uncased")
bert_out = nlp(idx, masks, segments)## fine-tuning

x = layers.GlobalAveragePooling1D()(bert_out[0])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
y_out = layers.Dense(7, activation='softmax')(x)## compile
model = models.Model([idx, masks, segments], y_out)

for layer in model.layers[:4]:
    layer.trainable = False
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

status = model.load_weights(filepath='bert-classification.h5')
with open('dicy.json', 'r') as fp:
    dic_y_mapping = json.load(fp)



def cpredict():
    corpus = e2.get()
    corpus = np.array([str(corpus)])
    maxlen = 50
    ## add special tokens
    maxqnans = np.int((maxlen-20)/2)
    corpus_tokenized = ["[CLS] "+
                " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', 
                str(txt).lower().strip()))[:maxqnans])+
                " [SEP] " for txt in corpus]

    ## generate masks
    masks = [[1]*len(txt.split(" ")) + [0]*(maxlen - len(
                txt.split(" "))) for txt in corpus_tokenized]
    ## padding
    txt2seq = [txt + " [PAD]"*(maxlen-len(txt.split(" "))) if len(txt.split(" ")) != maxlen else txt for txt in corpus_tokenized]

    ## generate idx
    idx = [tokenizer.encode(seq.split(" "), max_length=50, truncation=True) for seq in txt2seq]
    print(corpus_tokenized)
    ## generate segments
    segments = [] 
    for seq in txt2seq:
        temp, i = [], 0
        for token in seq.split(" "):
            temp.append(i)
            if token == "[SEP]":
                i += 1
        segments.append(temp)## feature matrix
    text = [np.asarray(idx, dtype='int32'), 
                np.asarray(masks, dtype='int32'), 
                np.asarray(segments, dtype='int32')]

    predicted_prob = model.predict(text)
    predicted = [dic_y_mapping[str(np.argmax(pred))] for pred in 
                predicted_prob]

    myText2.set(predicted[0])
    print(predicted[0])

def changeK():
    global k
    k = int(e3.get())

def summarize():
    def get_sentence_embeddings(sentence):
        embedding = model2.encode([sentence])
        return embedding[0]
    def distance_from_centroid(row):
        #type of emb and centroid is different, hence using tolist below
        return distance_matrix([row['embeddings']], [row['centroid'].tolist()])[0][0]
    article = e1.get(1.0, END)
    print(article)
    sentences=nltk.sent_tokenize(article)# strip leading and trailing spaces
    sentences = [sentence.strip() for sentence in sentences]
    print(sentences)
    data = pd.DataFrame(sentences)
    data.columns=['sentence']
    data['embeddings']=data['sentence'].apply(get_sentence_embeddings)
    global k
    NUM_CLUSTERS=k
    iterations=25
    X = np.array(data['embeddings'].tolist())
    kclusterer = KMeansClusterer(
            NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
            repeats=iterations,avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    data['cluster']=pd.Series(assigned_clusters, index=data.index)
    data['centroid']=data['cluster'].apply(lambda x: kclusterer.means()[x])
    data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)
    summary=' '.join(data.sort_values('distance_from_centroid',ascending = True).groupby('cluster').head(1).sort_index()['sentence'].tolist())
    result.insert(INSERT, summary)
 
master = Tk()
master.geometry("700x400")
myText=StringVar()
myText2=StringVar()

#row 0

Label(master, text="Text").grid(row=0, column=0, sticky=W, columnspan=3,padx=5,pady=5)

b = Button(master, text="Summarize", command=summarize)
b.grid(row=0, column=1,sticky=W+E+N+S, padx=5, pady=5)

#row 1

e1 = Text(master, width=60, height=20)
e1.grid(row=2, column=0)

#row 2

Label(master, text="Summarize to how many sentences?").grid(row=7, column=0, sticky=W, padx=5, pady=5)

e3 = Entry(master, width=60)
e3.grid(row=8, column=0, sticky=W, padx=5, pady=5)

b3 = Button(master, text="Submit", command=changeK)
b3.grid(row=8, column=1, sticky=W, padx=5, pady=5)

#row 3

Label(master, text="Summary:").grid(row=9, sticky=W)

#row 4

result=ScrolledText.ScrolledText(master, width=60, height=20, wrap=WORD)
result.grid(row=11,column=0)

#row 5

Label(master, text="Headline Classifier").grid(row=13, column=0, sticky=W,padx=5,pady=5)

b2 = Button(master, text="Classify", command=cpredict)
b2.grid(row=14, column=1, sticky=W, padx=5, pady=5)

#row 6

e2 = Entry(master, width=60)
e2.grid(row=14, column=0, padx=5, pady=5, sticky=W)

#row 7

Label(master, text="Category:").grid(row=16, column=0, sticky=W)
result2=Label(master, text="", textvariable=myText2, height=10).grid(row=17,column=0, sticky=W, padx=5, pady=5)

mainloop()