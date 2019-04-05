# -*- coding: utf-8 -*-

#generate negative link & remove some existed links for evalutation
from utils import load_data,load_nell,load_reddit
from scipy.sparse import csr_matrix
import numpy as np
import random
import copy
import pickle
#import time

#randomly remove existed links for LPG
def removed_links(adj,percent):
    total_num = adj.nnz/2
    remove_links_num = int(total_num*percent)
    removes = []
    
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    rows = adj_array.shape[0]
    changed_adj = copy.deepcopy(adj_array)
    
    # select the number of removed links
    print('The number of links will be removed is ',remove_links_num)
    for i in range(remove_links_num):
        while(True):
            row = random.randint(0,rows-1) #select a row randomly
            
            ones = []
            for col in range(len(changed_adj[row])):
                if changed_adj[row][col] == 1:
                    ones.append(col)
            
            #make sure there are no isolated nodes in graph
            nums = len(ones)
            if nums >1:
                col = ones[random.randint(0,nums-1)]
                if changed_adj[col].sum() > 1 and row!=col:
                    if changed_adj[row][col] == 0 or changed_adj[col][row] == 0:
                        print(row,'--',col)
                        print('remove error!!!')
                    changed_adj[row][col] = 0
                    changed_adj[col][row] = 0
                    removes.append([row,col])
                    break
    
    changed_adj = csr_matrix(changed_adj)
    removes = sorted(removes, key=lambda a:a[0])
    print('The number of removd links:', int(len(removes)) )
    print('The number links in changed adj',changed_adj.nnz/2)
    return changed_adj,removes
    

#generate negative links for training   
def negative_links(adj,pos_num):
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    neg_links = []
    for i in range(len(adj_array)):
        for j in range(i+1):
            if adj_array[i][j] == 0:
                neg_links.append([[i,j],-1])
    random.shuffle(neg_links)
    return neg_links[:pos_num]

#store postive links in a list for training
def positive_links(changed_array):
    changed_array = csr_matrix(changed_array, dtype=np.int8).toarray()
    pos_links = []
    for i in range(len(changed_array)):
        for j in range(i+1):
            if changed_array[i][j] == 1:
                pos_links.append([[i,j],1])
    return pos_links

#generate nagative test samples
def negative_test(pos_test_num,adj,neg_links):
    neg_test = []
    
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    rows = adj_array.shape[0]
    cols = adj_array.shape[1]
    
    for i in range(pos_test_num):
        while(True):
            row = random.randint(0,rows-1)
            col = random.randint(0,cols-1)
            
            if adj_array[row][col] == 0:
                if [[row,col],-1] not in neg_links and [[col,row],-1] not in neg_links and [row,col] not in neg_test and [col,row] not in neg_test:
                    neg_test.append([row,col])
                    break
                
    neg_test = sorted(neg_test, key=lambda a:a[0])
    print('The number of negative test links:', int(len(neg_test)))
    return neg_test

#remove setting
dataset = 'citeseer' # 'cora', 'citeseer', 'pubmed', 'nell.0.001', 'reddit'
remove_percentage = 0.1
if dataset == 'nell.0.001':
    adj = load_nell(dataset)[0]
elif dataset == 'reddit':
    adj = load_reddit()[0]
else:
    adj = load_data(dataset)[0]
#the number of non-zero
print('The number of links in original graph:',adj.nnz/2)

#remove links
changed_adj,removes = removed_links(adj,remove_percentage)

pos_links = positive_links(changed_adj)

neg_links = negative_links(adj,len(pos_links))

pos_links.extend(neg_links)
random.shuffle(pos_links)

neg_test = negative_test(len(removes),adj,neg_links)

#save links
links_path = dataset+'/links.pkl'
with open(links_path,'wb') as f:
    pickle.dump(pos_links,f)
adj_path = dataset+'/changed_adj.pkl'
with open(adj_path,'wb') as f:
    pickle.dump(changed_adj,f)
removes_path = dataset+'/pos_test.pkl'
with open(removes_path,'wb') as f:
    pickle.dump(removes,f)
negtest_path = dataset+'/neg_test.pkl'
with open(negtest_path,'wb') as f:
    pickle.dump(neg_test,f)  
