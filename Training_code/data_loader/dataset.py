import copy
import json
import logging
import sys
import os
os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
import dgl
import numpy as np
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug

type_0 = [1,7,9,12,16,17,21,22,27,33,34,35,37,40,47,48,55,56,59,63]
type_1 = [5,13,18,19,25,30,31,46,49,51,54,60,62,64,66,67,69]
node_type_0_set = set(type_0)
node_type_1_set = set(type_1)
##for each function
class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph =  dgl.heterograph({
        
    #Expression
    ('Expression', 'Expression0Expression', 'Expression'): ((),()),
    ('Expression', 'Expression1Expression', 'Expression'): ((),()),
    ('Expression', 'Expression2Expression', 'Expression'): ((),()),
    ('Expression', 'Expression3Expression', 'Expression'): ((),()),
    
    ('Expression', 'Expression0Function', 'Function'): ((),()),
    ('Expression', 'Expression1Function', 'Function'): ((),()),
    ('Expression', 'Expression2Function', 'Function'): ((),()),
    ('Expression', 'Expression3Function', 'Function'): ((),()),
    
    ('Expression', 'Expression0Statement', 'Statement'): ((),()),
    ('Expression', 'Expression1Statement', 'Statement'): ((),()),
    ('Expression', 'Expression2Statement', 'Statement'): ((),()),
    ('Expression', 'Expression3Statement', 'Statement'): ((),()),

    #Function
    ('Function', 'Function0Expression', 'Expression'): ((),()),
    ('Function', 'Function1Expression', 'Expression'): ((),()),
    ('Function', 'Function2Expression', 'Expression'): ((),()),
    ('Function', 'Function3Expression', 'Expression'): ((),()),
    
    ('Function', 'Function0Function', 'Function'): ((),()),
    ('Function', 'Function1Function', 'Function'): ((),()),
    ('Function', 'Function2Function', 'Function'): ((),()),
    ('Function', 'Function3Function', 'Function'): ((),()),
    
    ('Function', 'Function0Statement', 'Statement'): ((),()),
    ('Function', 'Function1Statement', 'Statement'): ((),()),
    ('Function', 'Function2Statement', 'Statement'): ((),()),
    ('Function', 'Function3Statement', 'Statement'): ((),()),
    
    #Statement
    ('Statement', 'Statement0Expression', 'Expression'): ((),()),
    ('Statement', 'Statement1Expression', 'Expression'): ((),()),
    ('Statement', 'Statement2Expression', 'Expression'): ((),()),
    ('Statement', 'Statement3Expression', 'Expression'): ((),()),
    
    ('Statement', 'Statement0Function', 'Function'): ((),()),
    ('Statement', 'Statement1Function', 'Function'): ((),()),
    ('Statement', 'Statement2Function', 'Function'): ((),()),
    ('Statement', 'Statement3Function', 'Function'): ((),()),
    
    ('Statement', 'Statement0Statement', 'Statement'): ((),()),
    ('Statement', 'Statement1Statement', 'Statement'): ((),()),
    ('Statement', 'Statement2Statement', 'Statement'): ((),()),
    ('Statement', 'Statement3Statement', 'Statement'): ((),()),
    
    
    })
        self.features = torch.FloatTensor(features)
        #self.graph.add_nodes(self.num_nodes, data={'features': self.features})   ##
        type_list = []
        ex_list = []
        st_list = []
        fu_list = []
        #print(self.features.shape)
        for i in range (0,self.features.shape[0]):
            features_new = self.features[i:i+1,:100]
            #print(features_new.shape)
            if(features[i][100] in node_type_0_set):
                type_new ='Expression'
                type_list.append(type_new)
                ex_list.append(i)
            elif(features[i][100] in node_type_1_set):
                type_new ='Statement'
                type_list.append(type_new)
                st_list.append(i)
            else:
                type_new='Function'
                type_list.append(type_new)
                fu_list.append(i)
            #print(features_type)
            self.graph = dgl.add_nodes(self.graph, num = 1, ntype = type_new,data = {'h':features_new})#
            #self.graph.nodes[features_type].data = torch.hstack((self.graph.nodes[features_type].data, features_new))
        #print(ex_list)
        #self.graph = dgl.add_nodes(self.graph, self.num_nodes, ntype = 'features')   ##
        #self.graph.nodes['features'].data['h'] = self.features
        #self.graph.add_nodes(self.num_nodes, data={self.features}, ntype = 'features')  ##
        #for i in range (0,self.num_nodes):
        #    self.graph.add_nodes(self.graph, 1, data=self.features[i], ntype='features')  ##
        
        
        
        for s, _type, t in edges:
            etype_number = str(self.dataset.get_edge_type_number(_type))
            #print(str(s)+" "+str(t)+" "+ str(etype_number))
            if etype_number == '4':
                continue
            '''
            if type_list[s]+etype_number+type_list[t] == 'Expression2Expression':
                continue
            if type_list[s]+etype_number+type_list[t] == 'Expression2Statement':
                continue
            if type_list[s]+etype_number+type_list[t] == 'Statement2Expression':
                continue
            if type_list[s]+etype_number+type_list[t] == 'Statement1Statement':
                continue
            '''
            
            for number in range (len(ex_list)):
                if s == ex_list[number]:
                    s_type = 'Expression'
                    s_number = number
                if t == ex_list[number]:
                    t_type = 'Expression'
                    t_number = number

            for number in range (len(st_list)):
                if s == st_list[number]:
                    s_type = 'Statement'
                    s_number = number
                if t == st_list[number]:
                    t_type = 'Statement'
                    t_number = number
                    
            for number in range (len(fu_list)):
                if s == fu_list[number]:
                    s_type = 'Function'
                    s_number = number
                if t == fu_list[number]:
                    t_type = 'Function'
                    t_number = number
            
            self.graph.add_edge(s_number, t_number, etype=(s_type, type_list[s]+etype_number+type_list[t], t_type))
            #self.graph.add_edge(t, s, etype=etype_number)
            #self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})  ##
        #print(self.graph)


class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()

    def initialize_dataset(self):

        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src, test_src):
        debug('Reading Train File!')
        #logging.info('train:' + train_src + '; valid:' + valid_src + '; test:' + test_src)
        
        with open(train_src, "r") as fp:
            train_data = []
            #for i in fp.readlines():
            #    train_data.append(json.loads(i))
            #for line in fp.readlines():
            #    train_data.append(json.loads(line))
            train_data = json.load(fp)
            for entry in tqdm(train_data):
                example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                                    
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)-1
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        '''

        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src, "r") as fp:
                valid_data = []
                #for i in fp.readlines():
                #    valid_data.append(json.loads(i))
                valid_data = json.load(fp) 
                for entry in tqdm(valid_data):
                    
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
                    
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src, "r") as fp:
                test_data = []
                #for i in fp.readlines():
                #    test_data.append(json.loads(i))
                test_data = json.load(fp) 
                for entry in tqdm(test_data):

                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.test_examples.append(example)
        '''

    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size

        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)


        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):

        #print(len(self.train_batches))
        if len(self.train_batches) == 0:
            #print('k'*40)
            self.initialize_train_batch()


        ids = self.train_batches.pop()
        if(len(self.train_batches) == 1):
            ids1 = self.train_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        if (len(self.valid_batches) == 1):
            ids1 = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        if (len(self.test_batches) == 1):
            ids1 = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
