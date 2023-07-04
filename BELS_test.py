
from BELS import *
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
"""
#for datasets less than 2,000 data items change the following items:

initial_examination = 50
initial_chunk = 100
otherwise:

initial_examination = 300
initial_chunk = 1000
"""
initial_examination = 300
initial_chunk = 1000
def generate_onehot_encoding(labels):
    encoder = OneHotEncoder(sparse=False)
    encoded_labels = encoder.fit_transform([[label] for label in labels])
    
    onehot_dict = {}
    for i, label in enumerate(labels):
        onehot_dict[str(int(label))] = encoded_labels[i]
    
    return onehot_dict
dataset_name= "phishing"
df = pd.read_csv(r""+dataset_name+".csv" )
dataset = np.array(df)
data = dataset[:,0:dataset.shape[1]-1]
labels = dataset[:,dataset.shape[1]-1]

unique_labels = np.unique(labels)
encodings = generate_onehot_encoding(unique_labels)

feature_size = data.shape[1]
label_size = len(np.unique(labels))
chunk_sizes = [2,5,10,20,50]

nodes= [[5,5,1],[5,5,50],[10,10,100]]

models = []
accs = []
print("Applying grid search step: 1/3...")
my_range = initial_examination / chunk_sizes[0]
my_range = int(my_range)
BELS_tmp_1 = BELS(False,initial_chunk, my_range,feature_size , label_size, chunk_sizes[0], nodes[0][0],nodes[0][1],
                nodes[0][2],True)
BELS_tmp_2 = BELS(False,initial_chunk, my_range,feature_size , label_size, chunk_sizes[0], nodes[0][0],nodes[0][1],
                nodes[0][2],False)

kappa,average_kappa,_, acc_1, _ = BELS_tmp_1.test_then_train(data, labels, encodings)
kappa,average_kappa,_, acc_2, _ = BELS_tmp_2.test_then_train(data, labels, encodings)

preprocess = True
if(acc_2-acc_1 > 5):
    preprocess = False
else:
    preprocess= True

flag = False

print("Applying grid search step: 2/3...")
for j in range(0,len(nodes)):
    print("Step: 2/3 - "+ str((j+1)) + "/"+ str(len(nodes)))
    for i in range(0, len(chunk_sizes)):
    
        my_range = initial_examination / chunk_sizes[i]
        acc = 0
        
        my_range = int(my_range)
        
        BELS_tmp = None
        BELS_tmp = BELS(flag,initial_chunk, my_range,feature_size , label_size, chunk_sizes[i], nodes[j][0],nodes[j][1],
                        nodes[j][2],preprocess)
        kappa,average_kappa,_, acc, _ = BELS_tmp.test_then_train(data, labels, encodings)
        
        accs.append(acc)
    
accs = np.array(accs)

max_value = np.max(accs)
indx=0
if(max_value == 100):
    indx=-1
indices = np.argwhere(accs == max_value).flatten()
accs_all = accs.copy()
max_index = indices[indx]


first_acc = accs[max_index]
accs[indices] = 0

max_value = np.max(accs)

indices = np.argwhere(accs == max_value).flatten()
max_index_2 = indices[indx]


accs[indices] = 0
nodes_index = int(max_index / len(chunk_sizes))
chunk_index = int(max_index % len(chunk_sizes))

print("Applying grid search 3/3...")
my_range = initial_examination / chunk_sizes[chunk_index]
my_range = int(my_range)
BELS_tmp = None
if(preprocess != False):
    BELS_tmp = BELS(flag,initial_chunk, my_range,feature_size , label_size, chunk_sizes[chunk_index], nodes[nodes_index][0]
                    ,nodes[nodes_index][1], nodes[nodes_index][2],False)
    kappa,average_kappa, _, acc, _ = BELS_tmp.test_then_train(data, labels, encodings)
else:
    acc = first_acc
flag = False


if(acc >= first_acc and preprocess != False):
    
    nodes_index_2 = int(max_index_2 / len(chunk_sizes))
    chunk_index_2 = int(max_index_2 % len(chunk_sizes))
    BELS_tmp = None
    my_range = initial_examination / chunk_sizes[chunk_index_2]
    my_range = int(my_range)
    BELS_tmp = BELS(flag,initial_chunk, my_range,feature_size , label_size, chunk_sizes[chunk_index_2], nodes[nodes_index_2][0]
                    ,nodes[nodes_index_2][1], nodes[nodes_index_2][2],False)
    kappa,average_kappa, _, acc_2, _ = BELS_tmp.test_then_train(data, labels, encodings)
    if(acc_2 > acc):
            nodes_index = nodes_index_2
            chunk_index = chunk_index_2

print("Parameters are finalized, the model starts the test then train phase: ")
time.sleep(3)

flag = True
preprocess= False
my_range = initial_chunk / chunk_sizes[chunk_index]
my_range = int(my_range)
total_time = 0
if(acc >= first_acc):
    
    BELS_tmp = BELS(flag,initial_chunk, my_range,feature_size , label_size, chunk_sizes[chunk_index], nodes[nodes_index][0]
                    ,nodes[nodes_index][1], nodes[nodes_index][2],preprocess)
    kappa,average_kappa,total_time, acc_final, batch_acc_list = BELS_tmp.test_then_train(data, labels, encodings)
else:
    preprocess= True
    
    BELS_tmp = BELS(flag, initial_chunk,my_range,feature_size , label_size, chunk_sizes[chunk_index], nodes[nodes_index][0]
                    ,nodes[nodes_index][1], nodes[nodes_index][2],preprocess)
    kappa,average_kappa,total_time, acc_final, batch_acc_list = BELS_tmp.test_then_train(data, labels, encodings)

np.save("1_"+dataset_name+"_kappa", kappa)
np.save("1_"+dataset_name+"_batch_acc", batch_acc_list)


print("runtime: ",total_time)
print("final accuracy: ",acc_final)
print("average kappa: ",average_kappa *100)
