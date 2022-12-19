import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from dgl.nn import GraphConv, AvgPooling, MaxPooling
from torch.autograd import Variable
import dgl
import dgl.function as fn
import math
import numpy as np
#from graph_transformer_edge_layer import GraphTransformerLayer

from mlp_readout import MLPReadout

from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm

# graph transformer
# HGT dgl version
class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.0, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        
        
        self.node_type_att = nn.Parameter(torch.ones(num_types))
        self.node_type_att1 = nn.Parameter(torch.ones(num_types))
        
        
        self.skip = nn.Parameter(torch.ones(num_types))
        
        self.weight = nn.Parameter(torch.ones(1))
        
        self.drop = nn.Dropout(dropout)
        
        self.attn_fc = nn.Linear(2*self.d_k, 1, bias=False)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        if(len(edges.data['id'])!= 0):
            
            etype = edges.data['id'][0]
            
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key = torch.bmm(edges.src['k'].transpose(1, 0).to(torch.device('cuda:0')), relation_att).transpose(1, 0)
            att = ((edges.dst['q'].to(torch.device('cuda:0')) * key).sum(dim=-1) * relation_pri / self.sqrt_dk)
            val = torch.bmm(edges.src['v'].transpose(1, 0).to(torch.device('cuda:0')), relation_msg).transpose(1, 0)
            
            if etype <= 9:
                src_ntype = 0
            elif etype <= 21:
                src_ntype = 1
            else: src_ntype = 2
            
            if etype <= 2 or (etype >= 10 and etype <=13) or (etype >= 22 and etype <=24):
                dst_ntype = 0
            elif (etype >=3 and etype <= 6) or (etype >= 14 and etype <=17) or (etype >= 25 and etype <=28):
                dst_ntype = 1
            else: dst_ntype = 2
            att_src = self.node_type_att[src_ntype]
            att_dst = self.node_type_att1[dst_ntype]
            att2 = torch.cat([att_dst * edges.dst['q'], att_src * edges.src['k']], dim=2)
            att2 = self.attn_fc(att2)
            att2 = att2.sum(dim=-1)  
            att2 = f.leaky_relu(att2)
        else:
            key = edges.src['k'].transpose(1, 0).to(torch.device('cuda:0')).transpose(1, 0) 
            att = ((edges.dst['q'].to(torch.device('cuda:0')) * key).sum(dim=-1)  / self.sqrt_dk) * 0
            val = (edges.src['v'].transpose(1, 0).to(torch.device('cuda:0'))).transpose(1, 0) 
            att2 = torch.cat([edges.dst['q'], edges.src['k']], dim=2)
            att2 = self.attn_fc(att2)
            att2 = att2.sum(dim=-1)  
            att2 = f.leaky_relu(att2) 

        

        return {'a': att, 'na': att2, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a'], 'na':edges.data['na']}

    def reduce_func(self, nodes):
        beta = torch.sigmoid(self.weight)
        att = f.softmax(nodes.mailbox['a'] + (beta) *nodes.mailbox['na'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        
        
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            #print(srctype)
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer='mean')                
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            #print(G.nodes[ntype].data['t'].to(torch.device('cuda:0')).shape)
            h = G.nodes[ntype].data['t'].to(torch.device('cuda:0')).shape[0]

            t = G.nodes[ntype].data['t'].to(torch.device('cuda:0'))
            trans_out = self.a_linears[n_id](t)
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key].to(torch.device('cuda:0')) * (1 - alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps

        self.gcs = nn.ModuleList()
        n_layers = 4
        n_heads = 4
        len_graph_ntypes = 3
        len_graph_etypes = 32
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        output_dim = 64
        for t in range(3):
            self.adapt_ws.append(nn.Linear(input_dim, output_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(output_dim, output_dim, len_graph_ntypes, len_graph_etypes, n_heads, use_norm=True))
        
        
        self.hidden_dim = 64
        self.batch_size = 256
        self.num_layers = 1
        self.bigru1 = nn.GRU(output_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.bigru2 = nn.GRU(output_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.bigru3 = nn.GRU(output_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.MPL_layer = MLPReadout(2*self.hidden_dim, 2)
        self.MPL_layer1 = MLPReadout(2*self.hidden_dim, 2)
        
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.sigmoid = nn.Sigmoid()
        
        self.weight = Variable(torch.ones(len_graph_ntypes).cuda())
        self.weight1 = Variable(torch.ones(len_graph_ntypes).cuda())

    def init_hidden(self):
        if True:
            if isinstance(self.bigru1, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            if isinstance(self.bigru2, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            if isinstance(self.bigru3, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num,  self.hidden_dim))
        return zeros.cuda()
        
    def forward(self, batch, cuda=False):
        graph = batch.get_network_inputs(cuda=cuda)
        
        graph = graph.to(torch.device('cuda:0'))
        #print(graph)
        graph.node_dict = {}
        graph.edge_dict = {}
        for ntype in graph.ntypes:
            graph.node_dict[ntype] = len(graph.node_dict)
            graph.nodes[ntype].data['id'] = torch.ones(graph.number_of_nodes(ntype), dtype=torch.long, device=graph.device) * graph.node_dict[ntype]
        for etype in graph.etypes:
            graph.edge_dict[etype] = len(graph.edge_dict)
            graph.edges[etype].data['id'] = torch.ones(graph.number_of_edges(etype), dtype=torch.long, device=graph.device) * graph.edge_dict[etype]
            
        for ntype in graph.ntypes:
            n_id = graph.node_dict[ntype]
            #self.adapt_ws[n_id](graph.nodes[ntype].data['h'].to(torch.device('cuda:0')))
            graph.nodes[ntype].data['new_h'] = torch.tanh(self.adapt_ws[n_id](graph.nodes[ntype].data['h'].to(torch.device('cuda:0'))))
            
            
            
        for i in range(self.n_layers):
            self.gcs[i](graph, 'new_h', 'new_h')
        #outputs = graph.nodes['Statement'].data['new_h']

        #node_Statement1 = dgl.readout_nodes(graph, 'new_h', op = 'sum', ntype = 'Statement')
        #node_Expression1 = dgl.readout_nodes(graph, 'new_h',  op = 'sum', ntype = 'Expression')
        #node_Function1 = dgl.readout_nodes(graph, 'new_h',  op = 'sum', ntype = 'Function')
        
        statement = graph.nodes['Statement'].data['new_h']
        expression = graph.nodes['Expression'].data['new_h']
        function = graph.nodes['Function'].data['new_h']
        st = graph.batch_num_nodes('Statement')
        ex = graph.batch_num_nodes('Expression')
        fu = graph.batch_num_nodes('Function')
        
        max_len_st = max(st)
        max_len_ex = max(ex)
        max_len_fu = max(fu)
        batch_size = len(st)
        st_seq, st_start, st_end = [], 0, 0
        ex_seq, ex_start, ex_end = [], 0, 0
        fu_seq, fu_start, fu_end = [], 0, 0
        for i in range (batch_size):
            st_end = st_start + st[i]
            ex_end = ex_start + ex[i]
            fu_end = fu_start + fu[i]
            if max_len_st - st[i]:
                st_seq.append(self.get_zeros(max_len_st-st[i]))
            if max_len_ex - ex[i]:
                ex_seq.append(self.get_zeros(max_len_ex-ex[i]))
            if max_len_fu - fu[i]:
                fu_seq.append(self.get_zeros(max_len_fu-fu[i]))
            st_seq.append(statement[st_start:st_end])
            ex_seq.append(expression[ex_start:ex_end])
            fu_seq.append(function[fu_start:fu_end])

            st_start = st_end
            ex_start = ex_end
            fu_start = fu_end

        st = torch.cat(st_seq)
        ex = torch.cat(ex_seq)
        fu = torch.cat(fu_seq)
        st = st.view(batch_size, max_len_st, -1)
        ex = ex.view(batch_size, max_len_ex, -1)
        fu = fu.view(batch_size, max_len_fu, -1)
        
        st, hidden = self.bigru1(st, self.hidden1)
        st = torch.transpose(st, 1, 2)
        ex, hidden = self.bigru2(ex, self.hidden2)
        ex = torch.transpose(ex, 1, 2)
        fu, hidden = self.bigru3(fu, self.hidden3)
        fu = torch.transpose(fu, 1, 2)        
        
        
        # pooling
        #print(st.shape)
        st1 = f.max_pool1d(st, st.size(2)).squeeze(2)
        ex1 = f.max_pool1d(ex, ex.size(2)).squeeze(2)
        fu1 = f.max_pool1d(fu, fu.size(2)).squeeze(2)
        
        st2 = f.avg_pool1d(st, st.size(2)).squeeze(2)
        ex2 = f.avg_pool1d(ex, ex.size(2)).squeeze(2)
        fu2 = f.avg_pool1d(fu, fu.size(2)).squeeze(2)
        #print(st.shape)
        '''
        all = st + ex + fu 

        st = st / all 
        ex = ex / all
        fu = fu / all
        
        length = len(st)
        st = st.reshape(length,1)
        ex = ex.reshape(length,1)
        fu = fu.reshape(length,1)
        outputs = st*node_Statement1 + ex * node_Expression1 + fu * node_Function1
        '''
        #print(outputs.shape)

        outputs = self.MPL_layer(self.weight[0] * st1+ self.weight[1] * ex1+self.weight[2] * fu1 + self.weight1[0] * st2+ self.weight1[1] * ex2+self.weight1[2] * fu2)
        


        outputs = nn.Softmax(dim=1)(outputs)
        #print(outputs)
        # outputs = avg.squeeze(dim = -1)
        return outputs






class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    #前向传播函数
    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result