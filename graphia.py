# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:40:54 2022

@author: fayya
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:36:27 2022

@author: u1876024
"""



from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
def writePyGGraph(G,ofname = 'temp.gml'):
    dict_coords = {'c'+str(i):G.coords[:,i] for i in range(G.coords.shape[1])}
    dict_feats = {'f'+str(i):G.x[:,i] for i in range(G.x.shape[1])}
    #import pdb;pdb.set_trace()
    dict_y = None# {'y'+str(i):G.y[:,i] for i in range(G.y.shape[1])}
    node_dict = {**dict_coords, **dict_feats}#,**dict_y
    d = Data(**node_dict,edge_index = G.edge_index, edge_attr = G.edge_attr)
    nG = to_networkx(d, node_attrs=list(node_dict.keys()))
    #nx.nx_pydot.write_dot(nG,'temp.dot')
    nx.write_gml(nG,ofname)

import numpy as np
import pickle
import torch
import io
class CPU_Unpickler(pickle.Unpickler):
    #https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
# with open(r'C:\Users\fayya\Downloads\TCGA-A2-A0T4.pkl', "rb") as f:    
#     G = CPU_Unpickler(f).load()
    
with open(r'C:\Users\fayya\Downloads\TCGA-A2-A0T4.pkl', "rb") as f:    
    G = torch.load(f)
G.x = G.x
"""
n1 = np.repeat(np.array([0,1,2,3,4,5,6]),5)
n2 = np.array([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4])
cat = np.stack((n1,n2), axis=0)
e = torch.tensor(cat, dtype=torch.long)
edge_index = e.t().clone().detach()
edge_attr = torch.tensor(np.random.rand(35,2))
x = torch.tensor([[0], [0], [0], [0], [0], [1], [1]], dtype=torch.float)
coords = torch.tensor(torch.randn((7,2)), dtype=torch.float)
feats = torch.tensor(torch.randn((7,4)), dtype=torch.float)

data = Data(x=feats,coords=coords, edge_index=edge_index.t().contiguous(), edge_attr = edge_attr)
print(data)
# Data(edge_attr=[35, 1], edge_index=[2, 35], x=[7, 1])

networkX_graph = to_networkx(data, node_attrs=["x","coords"], edge_attrs=["edge_attr"])

nx.nx_pydot.write_dot(networkX_graph,'temp.dot')

def write2dot(G):
    pass

G = data
"""
#G = G.to('cpu')
#d = Data(edge_index=G.edge_index.t().contiguous(), edge_attr = G.edge_attr)
# def writePyGGraph(G,ofname = 'temp.dot'):
#     dict_coords = {'c'+str(i):G.coords[:,i] for i in range(G.coords.shape[1])}
#     dict_feats = {'f'+str(i):G.x[:,i] for i in range(G.x.shape[1])}
#     dict_y = {'y'+str(i):G.y[:,i] for i in range(G.y.shape[1])}
#     node_dict = {**dict_coords, **dict_feats,**dict_y}
#     d = Data(**node_dict,edge_index = G.edge_index, edge_attr = G.edge_attr)
#     nG = to_networkx(d, node_attrs=list(node_dict.keys()))
#     nx.nx_pydot.write_dot(nG,'temp.dot')
writePyGGraph(G)