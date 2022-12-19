import torch
from dgl import DGLGraph

import numpy
import dgl
import dgl.function as fn
from torch import nn


class BatchGraph:
    def __init__(self):
        #self.graph = dgl.heterograph()#DGLGraph()
        self.graph = dgl.heterograph({
    #Expression
    ('Expression', 'Expression0Expression', 'Expression'): ((),()),
    ('Expression', 'Expression1Expression', 'Expression'): ((),()),
    #('Expression', 'Expression2Expression', 'Expression'): ((),()),
    ('Expression', 'Expression3Expression', 'Expression'): ((),()),
    
    ('Expression', 'Expression0Function', 'Function'): ((),()),
    ('Expression', 'Expression1Function', 'Function'): ((),()),
    ('Expression', 'Expression2Function', 'Function'): ((),()),
    ('Expression', 'Expression3Function', 'Function'): ((),()),
    
    ('Expression', 'Expression0Statement', 'Statement'): ((),()),
    ('Expression', 'Expression1Statement', 'Statement'): ((),()),
    #('Expression', 'Expression2Statement', 'Statement'): ((),()),
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
    #('Statement', 'Statement2Expression', 'Expression'): ((),()),
    ('Statement', 'Statement3Expression', 'Expression'): ((),()),
    
    ('Statement', 'Statement0Function', 'Function'): ((),()),
    ('Statement', 'Statement1Function', 'Function'): ((),()),
    ('Statement', 'Statement2Function', 'Function'): ((),()),
    ('Statement', 'Statement3Function', 'Function'): ((),()),
    
    ('Statement', 'Statement0Statement', 'Statement'): ((),()),
    #('Statement', 'Statement1Statement', 'Statement'): ((),()),
    ('Statement', 'Statement2Statement', 'Statement'): ((),()),
    ('Statement', 'Statement3Statement', 'Statement'): ((),()),
        })
        '''
    #Expression
    ('Expression', '0', 'Expression'): ((),()),
    ('Expression', '1', 'Expression'): ((),()),
    ('Expression', '2', 'Expression'): ((),()),
    ('Expression', '3', 'Expression'): ((),()),
    
    ('Expression', '0', 'Function'): ((),()),
    ('Expression', '1', 'Function'): ((),()),
    ('Expression', '2', 'Function'): ((),()),
    ('Expression', '3', 'Function'): ((),()),
    
    ('Expression', '0', 'Statement'): ((),()),
    ('Expression', '1', 'Statement'): ((),()),
    ('Expression', '2', 'Statement'): ((),()),
    ('Expression', '3', 'Statement'): ((),()),

    #Function
    ('Function', '0', 'Expression'): ((),()),
    ('Function', '1', 'Expression'): ((),()),
    ('Function', '2', 'Expression'): ((),()),
    ('Function', '3', 'Expression'): ((),()),
    
    ('Function', '0', 'Function'): ((),()),
    ('Function', '1', 'Function'): ((),()),
    ('Function', '2', 'Function'): ((),()),
    ('Function', '3', 'Function'): ((),()),
    
    ('Function', '0', 'Statement'): ((),()),
    ('Function', '1', 'Statement'): ((),()),
    ('Function', '2', 'Statement'): ((),()),
    ('Function', '3', 'Statement'): ((),()),
    
    #Statement
    ('Statement', '0', 'Expression'): ((),()),
    ('Statement', '1', 'Expression'): ((),()),
    ('Statement', '2', 'Expression'): ((),()),
    ('Statement', '3', 'Expression'): ((),()),
    
    ('Statement', '0', 'Function'): ((),()),
    ('Statement', '1', 'Function'): ((),()),
    ('Statement', '2', 'Function'): ((),()),
    ('Statement', '3', 'Function'): ((),())
    
    ('Statement', '0', 'Statement'): ((),()),
    ('Statement', '1', 'Statement'): ((),()),
    ('Statement', '2', 'Statement'): ((),()),
    ('Statement', '3', 'Statement'): ((),()),
    
    
    })
        '''
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0


    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
        num_edge_type = len(_g.canonical_etypes)
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))
        #self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        #self.graph.add_nodes(num_new_nodes, data={'features': self.features})
        #print(self.graph)
        #print(self.number_of_nodes)
        '''
        print(_g)

        for i in range (0,num_edge_type - 1):
            str_etype = str(i)
            sources, dests = _g.all_edges(etype = str_etype)
            self.graph.add_edge(sources, dests, etype= str_etype)


        sources += self.number_of_nodes
        dests += self.number_of_nodes
        #self.graph.add_edges(sources, dests, data=_g.edata)
        '''
        if(self.number_of_nodes == 0):
            self.graph = _g
        else:
            self.graph = dgl.batch([self.graph, _g])

        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1



    def cuda(self, device='cuda:0'):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)



    def de_batchify_graphs(self, features=None):
        print(self.graphid_to_nodeids.keys())
        '''
        assert isinstance(features, torch.Tensor)
        #print(features)
        #print(self.graphid_to_nodeids.keys())
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        #for i in self.graphid_to_nodeids.keys():
        #    vectors = features.index_select(dim=0, index=self.graphid_to_nodeids[gid])

        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            #print(v.device)
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors).to(torch.device('cuda:0'))
        #lengths = torch.LongTensor(lengths).to(device=output_vectors.device)
        '''
        return output_vectors#, lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')

from scipy import sparse as sp



class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()

    def get_network_inputs(self, cuda=False, device=None):
        #self.graph = dgl.add_self_loop(self.graph)
        #features = self.graph.ndata['features']
        #features = self.graph.nodes.data['h'].to(torch.device('cuda:0'))
        #图结构信息
        #edge_types = self.graph.edata['etype']
        if cuda:
            #self.cuda(device=device)
            return self.graph#, features#, edge_types#, _lap_pos_enc.cuda(device=device)
        else:
            return self.graph#, features, edge_types#, h_lap_pos_enc
        pass
