import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from agents.gnn import GATLayer, GCNLayer, SAGELayer
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import networkx as nx
import os

class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a # Number of actions.
        self.n_s = n_s # Number of states.
        self.n_step = n_step # Number of steps the policy should handle.
        self.identical = identical # A flag indicating whether all agents have identical policies.

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    '''Initializes the actor head with a fully connected layer'''
    def _init_actor_head(self, n_h, n_a=None): # n_h : the size of the hidden state passed to the critic head -> LSTM
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    '''
    Initializes the critic head with a fully connected layer
        n_h: The size of the hidden state (h), which comes from the preceding layers in the network (like the output of LSTM layers).
        n_n: Number of neighbors (relevant in multi-agent environments). If not provided, it defaults to the value of self.n_n. 
    '''
    def _init_critic_head(self, n_h, n_n=None): 
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a*n_n # each agent will have a sparse one-hot representation for its action.
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        # n_h as the input size and 1 as the output size (representing the value of the state or state-action pair).
        self.critic_head = nn.Linear(n_h, 1)  # V (s) / V(a,s)
        init_layer(self.critic_head, 'fc')

    '''
    Executes the critic head and computes value estimates based on the input hidden states.
        h: This is the hidden state vector (obtained from the output of LSTM or other layers).
        na: The actions taken by the agents. This can be either a batch of actions or a single action, depending on the context.
        n_n: The number of neighbors 
    '''
    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a*n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            # Move na_sparse to the same device as h
            na_sparse = na_sparse.to(h.device)
            h = torch.cat([h, na_sparse], dim=1)
        return self.critic_head(h).squeeze()

    #  Calculates the policy loss, value loss, and entropy loss
    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        '''action distribution'''
        log_probs = actor_dist.log_prob(As)
        '''policy loss'''
        policy_loss = -(log_probs * Advs).mean()
        '''entropy loss'''
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        '''value loss'''
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long().to(self.device),
                           torch.from_numpy(Rs).float().to(self.device),
                           torch.from_numpy(Advs).float().to(self.device))
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.device)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.device)
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().detach().cpu().numpy()
        else:
            return self._run_critic_head(h, np.array([naction])).detach().cpu().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_lstm * 2, device=self.device)
        self.states_bw = torch.zeros(self.n_lstm * 2, device=self.device)



class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        # Always use 128 as input size for LSTM
        self.lstm_layer = nn.LSTMCell(self.n_fc * 2, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        # Always pad to 128 features
        if x.size(1) < self.n_fc * 2:
            x = F.pad(x, (0, self.n_fc * 2 - x.size(1)))
        return x

class NCMultiAgentPolicy(Policy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime.

    Key Features:
        Centralized Meta-DNN:       All agents use a single shared model but process their inputs independently.
        Communication Mechanism:    Each agent communicates with its neighbors using a learned interaction model.
        Reinforcement Learning:     Implements actor-critic architecture to optimize policy and value functions.
    
    """
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        # Add device initialization at the start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        print("self.neighbor_mask=",self.neighbor_mask.sum(1))
        self.n_fc = n_fc
        self.n_h = n_h
        '''Initializes the communication and actor/critic networks using _init_net().'''
        self._init_net()

        self._reset()

    '''
    Calculates losses (policy, value, entropy) for each agent.
    Uses gradients to update the network parameters.
    '''
    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device) # obs: torch.Size([120, 25, 12])
        dones = torch.from_numpy(dones).float().to(self.device)             # dones: torch.Size([120])
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device) # fps: torch.Size([120, 25, 5]) 
        acts = torch.from_numpy(acts).long().to(self.device)                 # acts: torch.Size([25, 120])
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().to(self.device)
        Advs = torch.from_numpy(Advs).float().to(self.device)

        # Normalize advantages
        Advs = (Advs - Advs.mean()) / (Advs.std() + 1e-8)

        for i in range(self.n_agent):
            '''action distribution'''
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    '''Processes observations and optionally returns policy probabilities ('p') or value estimates.'''
    def forward(self, ob, done, fp, action=None, out_type='p'):
        # torch.Size([1, 25, 12])
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        # torch.Size([1])
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        # torch.Size([1, 25, 5])
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        
        # self.states_fw: torch.Size([25, 128]) is the hidden state of all agents
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw) #This returns the hidden states of all agents
        # h: torch.Size([25, 1,  64])
        # new_states: torch.Size([25, 128])
        if out_type.startswith('p'):
            # This generates the actor's policy (probability distribution over actions)
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            # This generates the critic's value estimate for the given action
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    '''
    Neighbor Communication 
        Computes the state of each agent by aggregating its own features, neighbor features, and messages from neighbors.
    '''
    def _get_comm_s(self, i, n_n, x, h, p):
        # x: obs torch.Size([25, 12])
        # Get the indices of the neighbors for agent i  
        # i=0, n_n=2, self.neighbor_mask[1]=tensor([1, 5])
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        # Get the hidden states of the neighbors
        # h: torch.Size([25, 64])

        # m_i: torch.Size([1, 128])
        # m_i is the hidden state of the neighbors
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        
        if self.identical:
            '''Get the observation of agent i  ''' 
            # x: torch.Size([1, 12])
            x_i = x[i].unsqueeze(0)
            '''Get the fingerprint of the policies of the neighbors'''
            # p: torch.Size([1, 10])  self.n_a=5
            p_i = torch.index_select(p, 0, js).view(1, self.n_a * n_n)
            '''Get the observation of the neighbors'''
            # nx_i: torch.Size([1, 24])  self.n_s=12
            nx_i = torch.index_select(x, 0, js).view(1, self.n_s * n_n)
        else:
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p[js[j]].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        
        '''Concatenate inputs and apply ReLU activation functions'''
        # s_i: [(x_i, nx_i), (p_i), (m_i)]
        x_combined = torch.cat([x_i, nx_i], dim=1)
        s_i = [F.relu(self.fc_x_layers[i](x_combined)),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        # s_i[MLP(obs_i, obs_neighbors), MLP(policy_neighbors), MLP(hidden_state_neighbors)]
        return torch.cat(s_i, dim=1)         # Concatenate the processed inputs along the last dimension

    def _get_neighbor_dim(self, i_agent):
        # Replace np.sum with torch.sum for PyTorch tensor
        n_n = int(torch.sum(self.neighbor_mask[i_agent]).item())
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            # Convert to numpy for iteration if needed
            mask = self.neighbor_mask[i_agent].cpu().numpy()
            for j in np.where(mask)[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        ''''
        n_n:    The number of neighbors for the current agent.
        n_ns:   Input dimension of neighbor states.
        n_na:   Input dimension of neighbor actions.

        fc_x_layer: Processes neighbor states                   (n_ns → self.n_fc).
        fc_p_layer: (Optional) Processes neighbor actions       (n_na → self.n_fc).
        fc_m_layer: (Optional) Processes aggregated messages    (self.n_h * n_n → self.n_fc).
        lstm_layer: Captures temporal dependencies, with a hidden size of self.n_h.
        '''
        n_lstm_in = 3 * self.n_fc # reflects the concatenation of multiple inputs (e.g., states, actions, or messages).
        ''' fc_x_layer: neighbor states '''
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n: # agent has neighbors
            ''' fc_p_layer: neighbor actions '''
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            ''' fc_m_layer: message aggregation '''
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc) # Processes messages from n_n neighbors, with each message having dimension self.n_h
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        '''
        self.n_h:   The size of the hidden state of the agent
        n_na:       The dimension of the agent's neighbor-related features. This represents additional information about the environment or other agents (e.g., communication messages or observations).
        '''
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)
    
    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=self.device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=self.device)

    def _run_actor_heads(self, hs, detach=False):
        # hs: TxNxh
        ps = []
        for i in range(self.n_agent):
            if detach:
                # This line outputs action probabilities, not a single action.
                # p_i is the policy of agent i | shape: [batch_size, n_actions]
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().cpu().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps # ps is a list of the policy of each agent

    '''
    Simulates communication between agents.
    Uses LSTM layers to process agent states, considering neighbor information.
    '''
    def _run_comm_layers(self, obs, dones, fps, states):
        # Convert all inputs to PyTorch tensors
        obs = torch.as_tensor(obs, device=self.device) # Observations from all agents
        dones = torch.as_tensor(dones, device=self.device) # Done flags for all agents
        fps = torch.as_tensor(fps, device=self.device) # Fingerprints of all agents 
        
        # Convert batch format to sequence format for LSTM processing
        # obs: torch.Size([1, 25, 12])
        obs = batch_to_seq(obs)         # Shape: [time_steps, n_agents, obs_dim]
        # dones: torch.Size([1，1])
        dones = batch_to_seq(dones)     
        # fps: torch.Size([1, 25, 5])
        fps = batch_to_seq(fps)         # Shape: [time_steps, n_agents, n_actions]
        
        # Split LSTM states into hidden state (h) and cell state (c)
        h, c = torch.chunk(states, 2, dim=1)
        
        outputs = []
        # Iterate over each time step and corresponding inputs
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0) # current observation | shape: [batch_size, obs_dim]
            p = p.squeeze(0) # current fingerprint | shape: [batch_size, n_agents, n_actions]
            # Iterate over each agent
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i] # number of neighbors for agent i
                if n_n:
                    # s_i [1,192] 3*64
                    s_i = self._get_comm_s(i, n_n, x, h, p) # shape: [1, obs_dim+policy_dim+hidden_state_dim]
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i_temp = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                        if self.unify_act_state_dim:
                            x_i = self.convert_state_linears[i](x_i_temp)
                        else:
                            x_i = x_i_temp
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                # Update hidden state and cell state for agent i
                # done flag is used to reset states
                h_i = h[i].unsqueeze(0) * (1-done)  # h_i is the hidden state of agent i | shape: [1, n_h]
                c_i = c[i].unsqueeze(0) * (1-done)  # c_i is the cell state of agent i | shape: [1, n_h]
                # LSTM layer updates
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h = torch.cat(next_h)
            c = torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        actions = actions.to(self.device)
        vs = []
        # Iterate over each agent
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i] # number of neighbors for agent i
            if n_n:
                # Move tensor to CPU before converting to numpy
                mask = self.neighbor_mask[i].cpu()
                # get the indices of the neighbors
                js = torch.nonzero(mask).squeeze(1).to(self.device)
                # Get the actions of the neighbors
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                # Iterate over each neighbor
                for j in range(n_n):
                    # Convert the action to one-hot encoding
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                # Concatenate the hidden state of agent i and the one-hot encoded actions of the neighbors
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze() # shape: [batch_size, 1]
            if detach:
                vs.append(v_i.detach().cpu().numpy())
            else:
                vs.append(v_i)
        return vs

class NCMultiAgentPolicy_MLP(Policy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime.

    Key Features:
        Centralized Meta-DNN:       All agents use a single shared model but process their inputs independently.
        Communication Mechanism:    Each agent communicates with its neighbors using a learned interaction model.
        Reinforcement Learning:     Implements actor-critic architecture to optimize policy and value functions.
    
    s_i[MLP(obs_i, obs_neighbors), MLP(policy_neighbors), MLP(hidden_state_neighbors)]

    """
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False):
        super(NCMultiAgentPolicy_MLP, self).__init__(n_a, n_s, n_step, 'nc_mlp', None, identical)
        # Add device initialization at the start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unify_act_state_dim = unify_act_state_dim

        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
            if self.unify_act_state_dim:
                # Initialize a list of Linear layers for each agent
                self.convert_state_shape = 16
                self.convert_state_linears = [self._create_linear(input_shape, self.convert_state_shape) for input_shape in n_s_ls]
                print(f"Created convert_state_linears: {[f'Linear({input_shape}, {self.convert_state_shape})' for input_shape in n_s_ls]}")
                
                self.convert_action_shape = max(n_a_ls)
                self.convert_action_linears = [self._create_linear(input_shape, self.convert_action_shape) for input_shape in n_a_ls]
                print(f"Created convert_action_linears: {[f'Linear({input_shape}, {self.convert_action_shape})' for input_shape in n_a_ls]}")
            
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        '''Initializes the communication and actor/critic networks using _init_net().'''
        self._init_net()

        self._reset()

    def _create_linear(self, input_shape,output_shape):
        # Define a single Linear layer to transform the input to a common shape
        layer = nn.Linear(input_shape, output_shape).to(self.device)
        print(f"Created linear layer: Linear({input_shape}, {output_shape})")
        return layer

    '''
    Calculates losses (policy, value, entropy) for each agent.
    Uses gradients to update the network parameters.
    '''
    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device) # obs: torch.Size([120, 25, 12])
        dones = torch.from_numpy(dones).float().to(self.device)             # dones: torch.Size([120])
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device) # fps: torch.Size([120, 25, 5]) 
        acts = torch.from_numpy(acts).long().to(self.device)                 # acts: torch.Size([25, 120])
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().to(self.device)
        Advs = torch.from_numpy(Advs).float().to(self.device)
        for i in range(self.n_agent):
            '''action distribution'''
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    '''Processes observations and optionally returns policy probabilities ('p') or value estimates.'''
    def forward(self, ob, done, fp, action=None, out_type='p'):
        # torch.Size([1, 25, 12])
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        # torch.Size([1])
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        # torch.Size([1, 25, 5])
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        
        # self.states_fw: torch.Size([25, 128]) is the hidden state of all agents
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw) #This returns the hidden states of all agents
        # h: torch.Size([25, 1,  64])
        # new_states: torch.Size([25, 128])
        if out_type.startswith('p'):
            # This generates the actor's policy (probability distribution over actions)
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            # This generates the critic's value estimate for the given action
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    '''
    Neighbor Communication 
        Computes the state of each agent by aggregating its own features, neighbor features, and messages from neighbors.
    '''
    def _get_comm_s(self, i, n_n, x, h, p):
        # x: obs torch.Size([25, 12])
        # Get the indices of the neighbors for agent i  
        # i=0, n_n=2, self.neighbor_mask[1]=tensor([1, 5])
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        # Get the hidden states of the neighbors
        # h: torch.Size([25, 64])

        # m_i: torch.Size([1, 128])
        # m_i is the hidden state of the neighbors
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        
        if self.identical:
            '''Get the observation of agent i  ''' 
            # x: torch.Size([1, 12])
            x_i = x[i].unsqueeze(0)
            '''Get the fingerprint of the policies of the neighbors'''
            # p: torch.Size([1, 10])  self.n_a=5
            p_i = torch.index_select(p, 0, js).view(1, self.n_a * n_n)
            '''Get the observation of the neighbors'''
            # nx_i: torch.Size([1, 24])  self.n_s=12
            nx_i = torch.index_select(x, 0, js).view(1, self.n_s * n_n)
        else:
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
            x_i = self.convert_state_linears[i](x_i)
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_temp = p[js[j]].narrow(0, 0, self.na_ls_ls[i][j])
                nx_i_temp = x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j])
                if self.unify_act_state_dim:
                    p_i_ls.append(self.convert_action_linears[js[j]](p_i_temp))
                    nx_i_ls.append(self.convert_state_linears[js[j]](nx_i_temp))
                else:
                    p_i_ls.append(p_i_temp)
                    nx_i_ls.append(nx_i_temp)
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        
        '''Concatenate inputs and apply ReLU activation functions'''
        # s_i: [(x_i, nx_i), (p_i), (m_i)]
        # s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))),
        #        F.relu(self.fc_p_layers[i](p_i)),
        #        F.relu(self.fc_m_layers[i](m_i))]
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        # s_i[MLP(obs_i, obs_neighbors), MLP(policy_neighbors), MLP(hidden_state_neighbors)]
        return torch.cat(s_i, dim=1)         # Concatenate the processed inputs along the last dimension

    def _get_neighbor_dim(self, i_agent):
        # Replace np.sum with torch.sum for PyTorch tensor
        n_n = int(torch.sum(self.neighbor_mask[i_agent]).item())
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            # Convert to numpy for iteration if needed
            mask = self.neighbor_mask[i_agent].cpu().numpy()
            for j in np.where(mask)[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.convert_state_shape * (n_n+1), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        print(f"Created actor head: Linear({self.n_h}, {n_a})")
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, convert_n_a):
        ''''
        n_n:    The number of neighbors for the current agent.
        n_ns:   Input dimension of neighbor states.
        convert_n_a: The dimension of the neighbor actions(max action dimension).

        fc_x_layer: Processes neighbor states                   (n_ns → self.n_fc).
        fc_p_layer: (Optional) Processes neighbor actions       (n_na → self.n_fc).
        fc_m_layer: (Optional) Processes aggregated messages    (self.n_h * n_n → self.n_fc).
        lstm_layer: Captures temporal dependencies, with a hidden size of self.n_h.
        '''
        n_lstm_in = 3 * self.n_fc # reflects the concatenation of multiple inputs (e.g., states, actions, or messages).
        ''' fc_x_layer: neighbor states '''
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        # print(f"Created fc_x_layer: Linear({n_ns}, {self.n_fc})")
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n: # agent has neighbors
            ''' fc_p_layer: neighbor actions '''
            fc_p_layer = nn.Linear(convert_n_a, self.n_fc)
            # print(f"Created fc_p_layer: Linear({convert_n_a}, {self.n_fc})")
            init_layer(fc_p_layer, 'fc')
            ''' fc_m_layer: message aggregation '''
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc) # Processes messages from n_n neighbors, with each message having dimension self.n_h
            # print(f"Created fc_m_layer: Linear({self.n_h * n_n}, {self.n_fc})")
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
            # print(f"Created lstm_layer: LSTMCell({n_lstm_in}, {self.n_h})")
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            # print(f"Created lstm_layer: LSTMCell({self.n_fc}, {self.n_h})")
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        '''
        self.n_h:   The size of the hidden state of the agent
        n_na:       The dimension of the agent's neighbor-related features. This represents additional information about the environment or other agents (e.g., communication messages or observations).
        '''
        critic_head = nn.Linear(self.n_h + n_na, 1)
        print(f"Created critic head: Linear({self.n_h + n_na}, 1)")
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)
    
    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            # convert the action dimension to the maximum action dimension
            convert_n_a = self.convert_action_shape * n_n

            self._init_comm_layer(n_n, n_ns, convert_n_a)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=self.device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=self.device)

    def _run_actor_heads(self, hs, detach=False):
        # hs: TxNxh
        ps = []
        for i in range(self.n_agent):
            if detach:
                # This line outputs action probabilities, not a single action.
                # p_i is the policy of agent i | shape: [batch_size, n_actions]
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().cpu().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps # ps is a list of the policy of each agent

    '''
    Simulates communication between agents.
    Uses LSTM layers to process agent states, considering neighbor information.
    '''
    def _run_comm_layers(self, obs, dones, fps, states):
        # Convert all inputs to PyTorch tensors
        obs = torch.as_tensor(obs, device=self.device) # Observations from all agents
        dones = torch.as_tensor(dones, device=self.device) # Done flags for all agents
        fps = torch.as_tensor(fps, device=self.device) # Fingerprints of all agents 
        
        # Convert batch format to sequence format for LSTM processing
        # obs: torch.Size([1, 25, 12])
        obs = batch_to_seq(obs)         # Shape: [time_steps, n_agents, obs_dim]
        # dones: torch.Size([1，1])
        dones = batch_to_seq(dones)     
        # fps: torch.Size([1, 25, 5])
        fps = batch_to_seq(fps)         # Shape: [time_steps, n_agents, n_actions]
        
        # Split LSTM states into hidden state (h) and cell state (c)
        h, c = torch.chunk(states, 2, dim=1)
        
        outputs = []
        # Iterate over each time step and corresponding inputs
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0) # current observation | shape: [batch_size, obs_dim]
            p = p.squeeze(0) # current fingerprint | shape: [batch_size, n_agents, n_actions]
            # Iterate over each agent
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i] # number of neighbors for agent i
                if n_n:
                    # s_i [1,192] 3*64
                    s_i = self._get_comm_s(i, n_n, x, h, p) # shape: [1, obs_dim+policy_dim+hidden_state_dim]
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i_temp = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                        if self.unify_act_state_dim:
                            x_i = self.convert_state_linears[i](x_i_temp)
                        else:
                            x_i = x_i_temp
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                # Update hidden state and cell state for agent i
                # done flag is used to reset states
                h_i = h[i].unsqueeze(0) * (1-done)  # h_i is the hidden state of agent i | shape: [1, n_h]
                c_i = c[i].unsqueeze(0) * (1-done)  # c_i is the cell state of agent i | shape: [1, n_h]
                # LSTM layer updates
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h = torch.cat(next_h)
            c = torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        actions = actions.to(self.device)
        vs = []
        # Iterate over each agent
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i] # number of neighbors for agent i
            if n_n:
                # Move tensor to CPU before converting to numpy
                mask = self.neighbor_mask[i].cpu()
                # get the indices of the neighbors
                js = torch.nonzero(mask).squeeze(1).to(self.device)
                # Get the actions of the neighbors
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                # Iterate over each neighbor
                for j in range(n_n):
                    # Convert the action to one-hot encoding
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                # Concatenate the hidden state of agent i and the one-hot encoded actions of the neighbors
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze() # shape: [batch_size, 1]
            if detach:
                vs.append(v_i.detach().cpu().numpy())
            else:
                vs.append(v_i)
        return vs

class GraphCMultiAgentPolicy(NCMultiAgentPolicy):
    '''
    obs, policy, hidden state has seperate GNN layer
    s_i[GNN(obs_i, obs_neighbors), GNN(policy_neighbors), GNN(hidden_state_neighbors)]
    '''
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False, n_heads=4, gnn_type='gat', use_random_mask=False):
        # Set device first before any tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.unify_act_state_dim = unify_act_state_dim
        # Store parameters needed before parent initialization
        self.neighbor_mask = neighbor_mask.to(self.device)
        print("neighbor_mask:",self.neighbor_mask.sum(axis=1))
        self.n_agent = n_agent
        self.n_fc = n_fc
        self.n_h = n_h
        self.n_heads = n_heads
        self.n_s = n_s
        self.n_a = n_a

        # Add random mask flag
        self.use_random_mask = use_random_mask
        
        # Store GNN type
        self.gnn_type = gnn_type.lower()
        
        # Initialize parent class
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'graph_nc', None, identical)
        
        if not self.identical:
            if n_s_ls is None or n_a_ls is None:
                raise ValueError("n_s_ls and n_a_ls must be provided for heterogeneous agents")
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
            if self.unify_act_state_dim:
                # Initialize a list of Linear layers for each agent
                self.convert_state_shape = 16
                self.convert_state_linears = [self._create_linear(input_shape, self.convert_state_shape) for input_shape in n_s_ls]
                print(f"Created convert_state_linears: {[f'Linear({input_shape}, {self.convert_state_shape})' for input_shape in n_s_ls]}")
                
                self.convert_action_shape = max(n_a_ls)
                self.convert_action_linears = [self._create_linear(input_shape, self.convert_action_shape) for input_shape in n_a_ls]
                print(f"Created convert_action_linears: {[f'Linear({input_shape}, {self.convert_action_shape})' for input_shape in n_a_ls]}")
        
        # Pre-compute neighbor dimensions
        self._precompute_neighbor_dims()
        
        # Initialize network components
        self._init_net()
        
        # Reset states
        self._reset()
        
        # Move model to device
        self.to(self.device)
    
    def _create_linear(self, input_shape,output_shape):
        # Define a single Linear layer to transform the input to a common shape
        layer = nn.Linear(input_shape, output_shape).to(self.device)
        print(f"Created linear layer: Linear({input_shape}, {output_shape})")
        return layer

    def _precompute_neighbor_dims(self):
        """Pre-compute neighbor dimensions for all agents"""
        self.neighbor_dims = []
        for i in range(self.n_agent):
            n_n = int(torch.sum(self.neighbor_mask[i]).item())
            if self.identical:
                dims = (n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n)
            else:
                mask = self.neighbor_mask[i].cpu().numpy()
                neighbor_indices = np.where(mask)[0]
                ns_ls = [self.n_s_ls[j] for j in neighbor_indices]
                na_ls = [self.n_a_ls[j] for j in neighbor_indices]
                dims = (n_n, self.n_s_ls[i] + sum(ns_ls), sum(na_ls), ns_ls, na_ls)
            self.neighbor_dims.append(dims)

    def _get_neighbor_dim(self, i_agent):
        return self.neighbor_dims[i_agent]

    def _init_gnn_layer(self, in_features, out_features, n_heads=None):
        """Initialize appropriate GNN layer based on gnn_type"""
        if self.gnn_type == 'gat':
            return GATLayer(in_features, out_features, n_heads or self.n_heads)
        elif self.gnn_type == 'gcn':
            return GCNLayer(in_features, out_features)
        elif self.gnn_type == 'sage':
            return SAGELayer(in_features, out_features)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def _run_comm_layers(self, obs, dones, fps, states):    
        # Convert all inputs to PyTorch tensors
        obs = torch.as_tensor(obs, device=self.device) # Observations from all agents
        dones = torch.as_tensor(dones, device=self.device) # Done flags for all agents
        fps = torch.as_tensor(fps, device=self.device) # Fingerprints of all agents 
        
        # Convert batch format to sequence format for LSTM processing
        # obs: torch.Size([1, 25, 12])
        obs = batch_to_seq(obs)         # Shape: [time_steps, n_agents, obs_dim]
        # dones: torch.Size([1，1])
        dones = batch_to_seq(dones)     
        # fps: torch.Size([1, 25, 5])
        fps = batch_to_seq(fps)         # Shape: [time_steps, n_agents, n_actions]
        
        # Split LSTM states into hidden state (h) and cell state (c)
        h, c = torch.chunk(states, 2, dim=1)
        
        outputs = []
        # Iterate over each time step and corresponding inputs
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0) # current observation | shape: [batch_size, obs_dim]
            p = p.squeeze(0) # current fingerprint | shape: [batch_size, n_agents, n_actions]
            # Iterate over each agent
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i] # number of neighbors for agent i
                if n_n:
                    '''if agent has neighbors
                       using GNN to process obs, policy, hidden state
                    '''
                    # s_i [1,192] 3*64
                    s_i = self._get_comm_s(i, n_n, x, h, p) # shape: [1, obs_dim+policy_dim+hidden_state_dim]
                else:
                    '''if agent has no neighbors
                       using MLP to process obs
                    '''
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                        # raise ValueError("Not implemented: No neighbors for identical agent i")
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                    s_i = F.relu(self.fc_x_layers_for_no_neighbor[i](x_i))
                # Update hidden state and cell state for agent i
                # done flag is used to reset states
                h_i = h[i].unsqueeze(0) * (1-done)  # h_i is the hidden state of agent i | shape: [1, n_h]
                c_i = c[i].unsqueeze(0) * (1-done)  # c_i is the cell state of agent i | shape: [1, n_h]
                # LSTM layer updates
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h = torch.cat(next_h)
            c = torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _get_comm_s(self, i, n_n, x, h, p):
        # Get neighbor indices once
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        h_i = h[js]   # [n_n, hidden_dim]
        
        if self.identical:
            # Batch all tensor operations
            x_i = x[i].unsqueeze(0)
            nx_i = x[js]  # [n_n, obs_dim]
            p_i = p[js]   # [n_n, action_dim]
            
        else:
            # Similar optimization for heterogeneous case
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)

            if self.unify_act_state_dim:
                x_i = self.convert_state_linears[i](x_i)
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_temp = p[js[j]].narrow(0, 0, self.na_ls_ls[i][j])
                nx_i_temp = x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j])
                if self.unify_act_state_dim:
                    p_i_ls.append(self.convert_action_linears[js[j]](p_i_temp))
                    nx_i_ls.append(self.convert_state_linears[js[j]](nx_i_temp))
                else:
                    p_i_ls.append(p_i_temp)
                    nx_i_ls.append(nx_i_temp)
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)

        # Create attention mask efficiently
        attn_mask = torch.zeros(n_n + 1, n_n + 1, device=self.device)
        
        if self.use_random_mask:
            # Generate random mask for neighbors
            random_mask = torch.bernoulli(torch.ones(n_n, device=self.device) * 0.5)  # 50% probability for each connection
            attn_mask[0, 1:] = attn_mask[1:, 0] = random_mask
        else:
            # Use original mask
            attn_mask[0, 1:] = attn_mask[1:, 0] = 1

        attn_mask +=  torch.eye(attn_mask.size(0), device=attn_mask.device)
            
        # Combine features efficiently
        x_combined = torch.cat([x_i, nx_i.reshape(n_n, -1)], dim=0)
        p_combined = torch.cat([p[i].unsqueeze(0), p_i.reshape(n_n, -1)], dim=0)
        h_combined = torch.cat([h[i].unsqueeze(0), h_i.reshape(n_n, -1)], dim=0)
        
        # Process all features in parallel
        return torch.cat([
                self.gnn_obs[i](x_combined, attn_mask)[0].unsqueeze(0),
                self.gnn_policy[i](p_combined, attn_mask)[0].unsqueeze(0),
                self.gnn_hidden[i](h_combined, attn_mask)[0].unsqueeze(0)
            ], dim=1)

    def _init_net(self):
        """Initialize network components efficiently"""
        # Initialize all ModuleLists at once
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        
        # Initialize GNN-specific ModuleLists
        self.gnn_obs = nn.ModuleList()
        self.gnn_policy = nn.ModuleList()
        self.gnn_hidden = nn.ModuleList()
        
        # Pre-allocate lists
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []

        self.fc_x_layers_for_no_neighbor = nn.ModuleList()
        
        # Initialize all components for each agent
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self.neighbor_dims[i]
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            
            self._init_comm_layer(n_n, n_ns, n_na,i)
            
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _init_comm_layer(self, n_n, n_ns, n_na, i_agent):
        '''Initialize communication layers with specified GNN type'''
        n_lstm_in = 3 * self.n_fc
        
        # Initialize GNN layers for observations
        if self.identical:
            gnn_obs = self._init_gnn_layer(self.n_s, self.n_fc)
        else:
            if self.unify_act_state_dim:
                gnn_obs = self._init_gnn_layer(self.convert_state_shape, self.n_fc)
            else:
                gnn_obs = self._init_gnn_layer(self.n_s_ls[i_agent], self.n_fc)

        self.gnn_obs.append(gnn_obs)

        if n_n: # If agent has neighbors
            # GNN layer for processing policies
            if self.identical:
                gnn_policy = self._init_gnn_layer(self.n_a, self.n_fc)
            else:
                if self.unify_act_state_dim:
                    gnn_policy = self._init_gnn_layer(self.convert_action_shape, self.n_fc)
                else:
                    gnn_policy = self._init_gnn_layer(self.n_a, self.n_fc)
            self.gnn_policy.append(gnn_policy)

            # GNN layer for processing hidden states
            gnn_hidden = self._init_gnn_layer(self.n_h, self.n_fc)
            self.gnn_hidden.append(gnn_hidden)
            
            self.fc_x_layers_for_no_neighbor.append(None)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.gnn_policy.append(None)
            self.gnn_hidden.append(None)
            if self.identical:
                no_neighbor_x_layer = nn.Linear(self.n_s, self.n_fc)
            else:
                no_neighbor_x_layer = nn.Linear(self.n_s_ls[i_agent], self.n_fc)
            self.fc_x_layers_for_no_neighbor.append(no_neighbor_x_layer)

            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)

        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)
    
    def _run_critic_heads(self, hs, actions, detach=False):
        actions = actions.to(self.device)
        vs = []
        # Iterate over each agent
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i] # number of neighbors for agent i
            if n_n:
                # Move tensor to CPU before converting to numpy
                mask = self.neighbor_mask[i].cpu()
                # get the indices of the neighbors
                js = torch.nonzero(mask).squeeze(1).to(self.device)
                # Get the actions of the neighbors
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                # Iterate over each neighbor
                for j in range(n_n):
                    # Convert the action to one-hot encoding
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                # Concatenate the hidden state of agent i and the one-hot encoded actions of the neighbors
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze() # shape: [batch_size, 1]
            if detach:
                vs.append(v_i.detach().cpu().numpy())
            else:
                vs.append(v_i)
        return vs

class GraphMaskGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphMaskGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s_i, s_j, temp=0.5, training=True):
        """
        s_i: Tensor of shape [n_n, d] or [1, d]
        s_j: Tensor of shape [n_n, d]
        """
        if s_i.shape[0] == 1:
            s_i = s_i.repeat(s_j.shape[0], 1)  # [n_n, d]

        x = torch.cat([s_i, s_j], dim=-1)  # [n_n, 2d]
        logits = self.fc2(F.relu(self.fc1(x)))  # [n_n, 1]
        prob = torch.sigmoid(logits)

        if training:
            u = torch.rand_like(prob)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            z = torch.sigmoid((logits + gumbel_noise) / temp)
            return z.squeeze(-1), prob.squeeze(-1)
        else:
            # During evaluation, use deterministic threshold
            mask = (prob > 0.08).float()
            
        return mask.squeeze(-1), prob.squeeze(-1)


class BayesianGraphCMultiAgentPolicy(GraphCMultiAgentPolicy):
    '''
    obs, policy, hidden state has seperate GNN layer
    s_i[GNN(obs_i, obs_neighbors, policy_neighbors, hidden_state_neighbors)]

    s_i[GNN(MLP(obs_i, obs_neighbors), MLP(policy_neighbors), MLP(hidden_state_neighbors))]
    '''
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False, n_heads=4, gnn_type='gat', learn_mask=True):
        # Convert unify_act_state_dim to boolean explicitly
        self.unify_act_state_dim = unify_act_state_dim
        self.learn_mask = learn_mask
        
        # Call parent class constructor
        super().__init__(n_s, n_a, n_agent, n_step, neighbor_mask, n_fc, n_h,
                         n_s_ls, n_a_ls, identical, unify_act_state_dim=self.unify_act_state_dim, 
                         n_heads=n_heads, gnn_type=gnn_type)
        
        self.n_heads = n_heads
        self.mask_history = []  # Store mask history for visualization
    
    def _init_comm_layer(self, n_n, n_ns, n_na, i_agent):
        '''Initialize communication layers with specified GNN type'''
        n_lstm_in = 3 * self.n_fc

        # === Initialize GNN layers for observations ===
        if self.identical:
            obs_input_dim = self.n_s
        elif self.unify_act_state_dim:
            obs_input_dim = self.convert_state_shape
        else:
            obs_input_dim = self.n_s_ls[i_agent]

        gnn_obs = self._init_gnn_layer(obs_input_dim, self.n_fc)
        self.gnn_obs.append(gnn_obs)

        # === Initialize GraphMaskGenerator if agent has neighbors ===
        if n_n:
            if not hasattr(self, 'mask_generators'):
                self.mask_generators = nn.ModuleList()

            # Each generator takes s_i and s_j as input (concatenated)
            mask_gen = GraphMaskGenerator(input_dim=2 * obs_input_dim, hidden_dim=self.n_fc)
            self.mask_generators.append(mask_gen)

            # === Policy GNN ===
            if self.identical:
                policy_input_dim = self.n_a
            elif self.unify_act_state_dim:
                policy_input_dim = self.convert_action_shape
            else:
                policy_input_dim = self.n_a
            gnn_policy = self._init_gnn_layer(policy_input_dim, self.n_fc)
            self.gnn_policy.append(gnn_policy)

            # === Hidden GNN ===
            gnn_hidden = self._init_gnn_layer(self.n_h, self.n_fc)
            self.gnn_hidden.append(gnn_hidden)

            # === LSTM ===
            self.fc_x_layers_for_no_neighbor.append(None)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)

        else:
            self.mask_generators.append(None)
            self.gnn_policy.append(None)
            self.gnn_hidden.append(None)

            if self.identical:
                no_neighbor_x_layer = nn.Linear(self.n_s, self.n_fc)
            else:
                no_neighbor_x_layer = nn.Linear(self.n_s_ls[i_agent], self.n_fc)
            self.fc_x_layers_for_no_neighbor.append(no_neighbor_x_layer)

            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)

        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)
    
    def _get_comm_s(self, i, n_n, x, h, p):
        # Get neighbor indices
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        h_i = h[js]   # [n_n, hidden_dim]

        # === Feature Gathering ===
        if self.identical:
            x_i = x[i].unsqueeze(0)       # shape: [1, obs_dim]
            nx_i = x[js]                  # [n_n, obs_dim]
            p_i = p[js]                   # [n_n, action_dim]
        else:
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
            if self.unify_act_state_dim:
                x_i = self.convert_state_linears[i](x_i)

            p_i_ls, nx_i_ls = [], []
            for j in range(n_n):
                p_temp = p[js[j]].narrow(0, 0, self.na_ls_ls[i][j])
                x_temp = x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j])
                if self.unify_act_state_dim:
                    p_i_ls.append(self.convert_action_linears[js[j]](p_temp))
                    nx_i_ls.append(self.convert_state_linears[js[j]](x_temp))
                else:
                    p_i_ls.append(p_temp)
                    nx_i_ls.append(x_temp)
            p_i = torch.cat(p_i_ls).unsqueeze(0)       # [1, total_action_dim]
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)     # [1, total_state_dim]

        # === Create attention mask ===
        attn_mask = torch.zeros(n_n + 1, n_n + 1, device=self.device)
        
        if self.learn_mask and self.mask_generators[i] is not None:
            # === Sample Z_t from GraphMaskGenerator ===
            mask_gen = self.mask_generators[i]
            s_i_state = x_i                        # [1, d]
            s_j_states = nx_i.view(n_n, -1)       # [n_n, d]

            z_mask, prob_mask = mask_gen(s_i_state, s_j_states, training=self.training)  # [n_n], [n_n]

            # Save for ELBO loss
            if not hasattr(self, 'latest_mask_probs'):
                self.latest_mask_probs = {}
            self.latest_mask_probs[i] = prob_mask  # detach if needed

            # Apply learned mask
            attn_mask[0, 1:] = attn_mask[1:, 0] = z_mask
        else:
            # Use full mask (all connections enabled)
            attn_mask[0, 1:] = attn_mask[1:, 0] = 1

        # Add self-loops to make sure nodes always attend to themselves
        attn_mask += torch.eye(n_n + 1, device=self.device)

        # === Combine inputs for GNN ===
        x_combined = torch.cat([x_i, nx_i.reshape(n_n, -1)], dim=0)
        p_combined = torch.cat([p[i].unsqueeze(0), p_i.reshape(n_n, -1)], dim=0)
        h_combined = torch.cat([h[i].unsqueeze(0), h_i.reshape(n_n, -1)], dim=0)

        # === GNN feature aggregation ===
        return torch.cat([
            self.gnn_obs[i](x_combined, attn_mask)[0].unsqueeze(0),
            self.gnn_policy[i](p_combined, attn_mask)[0].unsqueeze(0),
            self.gnn_hidden[i](h_combined, attn_mask)[0].unsqueeze(0)
        ], dim=1)
    
    def forward(self, ob, done, fp, action=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()

        # Temporarily switch to eval mode for masking
        training_backup = self.training
        self.eval()

        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)

        # Restore training flag
        if training_backup:
            self.train()

        self.states_fw = new_states.detach()

         # Store mask history for visualization
        if hasattr(self, 'latest_mask_probs'):
            self.mask_history.append({
                'step': getattr(self, 'current_step', 0),
                'masks': {i: p.detach().cpu().numpy() 
                         for i, p in self.latest_mask_probs.items()}
            })

        if out_type.startswith('p'):
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    def backward(self, obs, fps, acts, dones, Rs, Advs,
             e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device)
        acts = torch.from_numpy(acts).long().to(self.device)

        # === Forward pass through comm layers ===
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        self.states_bw = new_states.detach()

        # Actor and critic outputs
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)

        # Rewards and advantages
        Rs = torch.from_numpy(Rs).float().to(self.device)
        Advs = torch.from_numpy(Advs).float().to(self.device)

        # Normalize advantages
        Advs = (Advs - Advs.mean()) / (Advs.std() + 1e-8)


        # === Loss components ===
        likelihood_term = 0
        value_term = 0
        entropy_term = 0
        prior_term = 0
        mask_entropy_term = 0

        for i in range(self.n_agent):
            actor_dist = torch.distributions.categorical.Categorical(logits=ps[i])

            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist, e_coef, v_coef, vs[i], acts[i], Rs[i], Advs[i])

            # Likelihood = - policy loss
            likelihood_term += -policy_loss_i
            value_term += value_loss_i
            entropy_term += entropy_loss_i

            # === ELBO-specific terms ===
            if hasattr(self, 'latest_mask_probs') and i in self.latest_mask_probs:
                mask_probs = self.latest_mask_probs[i]

                # Bernoulli prior log-prob: log p(Z_t)
                lambda_ = getattr(self, 'lambda_', 0.2)
                prior_term += (lambda_ * torch.log(mask_probs + 1e-8) +
                            (1 - lambda_) * torch.log(1 - mask_probs + 1e-8)).sum()

                # Entropy of q(Z_t; φ): -log q
                mask_entropy = - (mask_probs * torch.log(mask_probs + 1e-8) +
                                (1 - mask_probs) * torch.log(1 - mask_probs + 1e-8))
                mask_entropy_term += mask_entropy.sum()

        # === Final ELBO loss ===
        self.policy_loss = -likelihood_term
        self.value_loss = value_term
        self.entropy_loss = entropy_term
        self.prior_loss = prior_term
        self.mask_entropy_loss = mask_entropy_term
        self.elbo_loss = -(likelihood_term + prior_term - mask_entropy_term)

        self.loss = self.elbo_loss + self.value_loss + self.entropy_loss
        self.loss.backward()

        # Logging (optional)
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_prior_loss'.format(self.name), self.prior_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_mask_entropy_loss'.format(self.name), self.mask_entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_elbo_loss'.format(self.name), self.elbo_loss,
                                  global_step=global_step)
    
    def visualize_masks(self, step, save_path=None, draw_whole=False):
        """Visualize the learned attention masks for each agent
        Args:
            step: Current environment step
            save_path: Optional path to save the visualization
            draw_whole: If True, shows only the whole graph visualization
        """
        if not hasattr(self, 'latest_mask_probs'):
            logging.warning("No mask probabilities available for visualization")
            return

        import networkx as nx
        import os
        
        if draw_whole:
            # Create a single figure for the whole graph
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Store all selected edges
            selected_edges = []
            edge_probs = []
            
            # Collect all selected edges and their probabilities
            for i in range(self.n_agent):
                if i in self.latest_mask_probs:
                    js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
                    probs = self.latest_mask_probs[i].detach().cpu().numpy()
                    for idx, j in enumerate(js):
                        if probs[idx] > 0.08:  # Using the same threshold as in the model
                            selected_edges.append((i, j.item()))
                            edge_probs.append(float(probs[idx]))  # Convert to Python float
            
            # Create the adjacency matrix
            adj_matrix = torch.zeros((self.n_agent, self.n_agent), dtype=torch.float32)
            for (i, j), prob in zip(selected_edges, edge_probs):
                adj_matrix[i, j] = float(prob)  # Ensure float type
                adj_matrix[j, i] = float(prob)  # Make it symmetric
            
            # Plot adjacency matrix heatmap
            sns.heatmap(adj_matrix.cpu().numpy(), ax=ax1,
                       cmap='YlOrRd',  # Yellow-Orange-Red colormap to match edge color
                       vmin=0, vmax=0.5,
                       cbar_kws={'label': 'Connection Probability'})
            # Adjust colorbar label font size after creation
            ax1.collections[0].colorbar.ax.set_ylabel('Connection Probability', fontsize=16)
            ax1.set_title('Whole Graph Adjacency Matrix', fontsize=18)
            
            # Plot grid network
            G = nx.from_numpy_array(adj_matrix.cpu().numpy())
            
            # Calculate grid dimensions
            grid_size = int(np.sqrt(self.n_agent))
            if grid_size * grid_size != self.n_agent:
                grid_size = int(np.ceil(np.sqrt(self.n_agent)))
            
            # Create a proper grid layout
            pos = {}
            for j in range(self.n_agent):
                row = j // grid_size
                col = j % grid_size
                pos[j] = (col, -row)  # x=column, y=-row (negative to have 0,0 at top-left)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax2,
                                 node_color='#1F77B4',
                                 node_size=800)
            
            # Draw edges with varying widths based on probability
            for (i, j), prob in zip(selected_edges, edge_probs):
                nx.draw_networkx_edges(G, pos, ax=ax2,
                                     edgelist=[(i, j)],
                                     edge_color='#E46D4C',
                                     width=float(prob) * 10)  # Ensure float type
            
            # Add labels with font customization
            nx.draw_networkx_labels(G, pos, ax=ax2, font_size=18, font_color='white')
            ax2.set_title('Whole Grid Network\n(Edge width indicates probability)', fontsize=18)
            ax2.axis('off')
            
            plt.suptitle(f'Whole Graph Visualization at Step {step}', fontsize=18)
            plt.tight_layout()
            
        else:
            # Original detailed per-agent visualization
            n_agents = self.n_agent
            n_cols = 4  # Local mask probs, Local adj, Global adj, Grid network
            fig, axes = plt.subplots(n_agents, n_cols, figsize=(6*n_cols, 4*n_agents))
            if n_agents == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(n_agents):
                if i in self.latest_mask_probs:
                    # Get 1st hop neighbors
                    js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
                    n_neighbors = len(js)
                    
                    # 1. Plot local mask probabilities
                    probs = self.latest_mask_probs[i].detach().cpu().numpy()
                    logging.info(f"Agent {i} mask probabilities: {probs}")
                    
                    sns.barplot(x=range(n_neighbors), y=probs, ax=axes[i, 0])
                    axes[i, 0].set_title(f'Agent {i} Local Mask Probabilities')
                    axes[i, 0].set_xlabel('Neighbor Index')
                    axes[i, 0].set_ylabel('Probability')
                    
                    # 2. Plot local adjacency matrix (1st hop)
                    local_mask = torch.zeros((n_neighbors + 1, n_neighbors + 1), device=self.device, dtype=torch.float32)
                    # Add self-connection
                    local_mask[0, 0] = 1.0
                    # Add connections to neighbors based on mask probabilities
                    local_mask[0, 1:] = torch.tensor((probs > 0.1).astype(np.float32), device=self.device)
                    local_mask[1:, 0] = local_mask[0, 1:]  # Make it symmetric
                    
                    # Create labels for the local adjacency matrix
                    labels = ['self'] + [f'n{j.item()}' for j in js]
                    
                    sns.heatmap(local_mask.cpu().numpy(), ax=axes[i, 1], 
                               cmap='YlOrRd', vmin=0, vmax=1,
                               xticklabels=labels, yticklabels=labels)
                    axes[i, 1].set_title(f'Agent {i} Local Adjacency')
                    
                    # 3. Plot global adjacency matrix with highlighted local connections
                    global_mask = self.neighbor_mask.clone().float()
                    # Highlight local connections in a different color
                    highlight_mask = torch.zeros_like(global_mask)
                    highlight_mask[i, js] = torch.tensor((probs > 0.1).astype(np.float32), device=self.device)
                    
                    # Create a custom colormap that highlights local connections
                    sns.heatmap(global_mask.cpu().numpy(), ax=axes[i, 2],
                               cmap='Blues', vmin=0, vmax=1, alpha=0.5)
                    sns.heatmap(highlight_mask.cpu().numpy(), ax=axes[i, 2],
                               cmap='Reds', vmin=0, vmax=1, alpha=0.7)
                    axes[i, 2].set_title(f'Agent {i} Global Adjacency\n(Red=Selected Local, Blue=Global)')
                    
                    # 4. Plot grid network visualization
                    G = nx.from_numpy_array(global_mask.cpu().numpy())
                    
                    # Calculate grid dimensions
                    grid_size = int(np.sqrt(n_agents))
                    if grid_size * grid_size != n_agents:
                        grid_size = int(np.ceil(np.sqrt(n_agents)))
                    
                    # Create a proper grid layout
                    pos = {}
                    for j in range(n_agents):
                        row = j // grid_size
                        col = j % grid_size
                        pos[j] = (col, -row)  # x=column, y=-row (negative to have 0,0 at top-left)
                    
                    # Draw the base grid
                    nx.draw_networkx_nodes(G, pos, ax=axes[i, 3], 
                                         node_color='lightblue', 
                                         node_size=500)
                    nx.draw_networkx_edges(G, pos, ax=axes[i, 3],
                                         edge_color='gray',
                                         alpha=0.5)
                    
                    # Highlight selected local connections with edge weights
                    for idx, j in enumerate(js):
                        if probs[idx] > 0.1:  # Using the same threshold as in the model
                            nx.draw_networkx_edges(
                                G, pos, ax=axes[i, 3],
                                edgelist=[(i, j.item())],
                                edge_color='red',
                                width=float(probs[idx]) * 5  # Ensure float type
                            )
                    
                    # Highlight the current agent
                    nx.draw_networkx_nodes(G, pos, ax=axes[i, 3],
                                         nodelist=[i],
                                         node_color='red',
                                         node_size=500)
                    
                    # Add labels
                    nx.draw_networkx_labels(G, pos, ax=axes[i, 3])
                    axes[i, 3].set_title(f'Agent {i} Grid Network\n(Red=Selected, Blue=Others)')
                    axes[i, 3].axis('off')  # Hide axes
                    
            plt.suptitle(f'Mask Visualization at Step {step}\nLocal vs Global Connectivity')
            plt.tight_layout()
            
        if save_path:
            try:
                # Ensure the directory exists
                os.makedirs(save_path, exist_ok=True)
                
                # Create the full file path
                save_file = os.path.join(save_path, f'mask_step_{step}.png')
                
                # Save the figure
                plt.savefig(save_file, dpi=500, bbox_inches='tight')
                logging.info(f"Successfully saved visualization to {save_file}")
                
                # Verify the file was created
                if os.path.exists(save_file):
                    file_size = os.path.getsize(save_file)
                    logging.info(f"File created successfully. Size: {file_size} bytes")
                else:
                    logging.error(f"File was not created at {save_file}")
                
            except Exception as e:
                logging.error(f"Error saving visualization: {str(e)}")
                logging.error(f"Save path: {save_path}")
                logging.error(f"Current working directory: {os.getcwd()}")
            finally:
                plt.close()
        else:
            plt.show()


class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
        
        # Add gradient clipping
        self.max_grad_norm = 0.5

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            
            # Initialize layers with proper weight initialization
            fc_x_layer = nn.Linear(n_s, self.n_fc)
            nn.init.orthogonal_(fc_x_layer.weight, gain=np.sqrt(2))
            nn.init.constant_(fc_x_layer.bias, 0)
            self.fc_x_layers.append(fc_x_layer)
            
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            for name, param in lstm_layer.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            self.lstm_layers.append(lstm_layer)
            
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(torch.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        # NxTxm
        obs = obs.transpose(0, 1).to(self.device) # [28,1,22]
        dones = torch.as_tensor(dones, device=self.device)
        hs = []
        new_states = []
        if self.identical:
            for i in range(self.n_agent):
                xs_i = F.relu(self.fc_x_layers[i](obs[i]))
                hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
                hs.append(hs_i.unsqueeze(0))
                new_states.append(new_states_i.unsqueeze(0))
        else:
            obs_dim = self.n_s_ls
            for i in range(self.n_agent):
                xs_i = F.relu(self.fc_x_layers[i](obs[i][:, :obs_dim[i]]))
                hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
                hs.append(hs_i.unsqueeze(0))
                new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device)
        acts = torch.from_numpy(acts).long().to(self.device)
        
        # Add numerical stability checks
        # Replace NaN values with 0 in observation and fps tensors
        obs = torch.nan_to_num(obs, nan=0.0)
        fps = torch.nan_to_num(fps, nan=0.0)
            
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        self.states_bw = new_states.detach()
        
        # Add stability check for hidden states
        # if torch.isnan(hs).any():
        #     raise ValueError("NaN values detected in hidden states")
            
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        
        # Add stability check for policy logits - handle list of tensors
        for i, p in enumerate(ps):
            if torch.isnan(p).any():
                # Replace NaN values with -1000000 to make those actions have near-zero probability
                ps[i] = torch.nan_to_num(p, nan=-1000000.0)
                # Convert logits to probabilities using softmax
                ps[i] = F.softmax(ps[i], dim=-1)  # Apply softmax for numerical stability
            
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().to(self.device)
        Advs = torch.from_numpy(Advs).float().to(self.device)

        # Normalize advantages with numerical stability
        Advs = (Advs - Advs.mean()) / (Advs.std() + 1e-8)

        for i in range(self.n_agent):
            # Add stability check for action distribution
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            if torch.isnan(actor_dist_i.logits).any():
                raise ValueError(f"NaN values detected in action distribution for agent {i}")
                
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
            
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        
        # Add gradient clipping
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs.
       s_i=[MLP(obs_i, obs_neighbors)+MLP(hidden_state_neighbors)]"""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unify_act_state_dim = unify_act_state_dim


        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
            if self.unify_act_state_dim:
                # Initialize a list of Linear layers for each agent
                self.convert_state_shape = 16
                self.convert_state_linears = [self._create_linear(input_shape, self.convert_state_shape) for input_shape in n_s_ls]
                print(f"Created convert_state_linears: {[f'Linear({input_shape}, {self.convert_state_shape})' for input_shape in n_s_ls]}")
                
                self.convert_action_shape = max(n_a_ls)
                self.convert_action_linears = [self._create_linear(input_shape, self.convert_action_shape) for input_shape in n_a_ls]
                print(f"Created convert_action_linears: {[f'Linear({input_shape}, {self.convert_action_shape})' for input_shape in n_a_ls]}")

        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()
    
    def _create_linear(self, input_shape,output_shape):
        # Define a single Linear layer to transform the input to a common shape
        layer = nn.Linear(input_shape, output_shape).to(self.device)
        # print(f"Created linear layer: Linear({input_shape}, {output_shape})")
        return layer
    
    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            if self.unify_act_state_dim:
                n_ns = self.convert_state_shape * (n_n + 1)
                # n_na = self.convert_action_shape 
            self._init_comm_layer(n_n, n_ns)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _init_comm_layer(self, n_n, n_ns):
        '''
        n_n:    The number of neighbors for the current agent.
        n_ns:   Input dimension of neighbor states.
        '''
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)
    
        '''
    Simulates communication between agents.
    Uses LSTM layers to process agent states, considering neighbor information.
    '''
    def _run_comm_layers(self, obs, dones, fps, states):
        # Convert all inputs to PyTorch tensors
        obs = torch.as_tensor(obs, device=self.device) # Observations from all agents
        dones = torch.as_tensor(dones, device=self.device) # Done flags for all agents
        fps = torch.as_tensor(fps, device=self.device) # Fingerprints of all agents 
        
        # Convert batch format to sequence format for LSTM processing
        # obs: torch.Size([1, 25, 12])
        obs = batch_to_seq(obs)         # Shape: [time_steps, n_agents, obs_dim]
        # dones: torch.Size([1，1])
        dones = batch_to_seq(dones)     
        # fps: torch.Size([1, 25, 5])
        fps = batch_to_seq(fps)         # Shape: [time_steps, n_agents, n_actions]
        
        # Split LSTM states into hidden state (h) and cell state (c)
        h, c = torch.chunk(states, 2, dim=1)
        
        outputs = []
        # Iterate over each time step and corresponding inputs
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0) # current observation | shape: [batch_size, obs_dim]
            p = p.squeeze(0) # current fingerprint | shape: [batch_size, n_agents, n_actions]
            # Iterate over each agent
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i] # number of neighbors for agent i
                if n_n:
                    # s_i [1,192] 3*64
                    s_i = self._get_comm_s(i, n_n, x, h, p) # shape: [1, obs_dim+policy_dim+hidden_state_dim]
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i_temp = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                        if self.unify_act_state_dim:
                            x_i = self.convert_state_linears[i](x_i_temp)
                        else:
                            x_i = x_i_temp
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                # Update hidden state and cell state for agent i
                # done flag is used to reset states
                h_i = h[i].unsqueeze(0) * (1-done)  # h_i is the hidden state of agent i | shape: [1, n_h]
                c_i = c[i].unsqueeze(0) * (1-done)  # c_i is the cell state of agent i | shape: [1, n_h]
                # LSTM layer updates
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h = torch.cat(next_h)
            c = torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _get_comm_s(self, i, n_n, x, h, p):
        # js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
            if self.unify_act_state_dim:
                x_i = self.convert_state_linears[i](x_i)

            nx_i_ls = []
            for j in range(n_n):
                nx_i_temp = x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j])
                if self.unify_act_state_dim:
                    nx_i_ls.append(self.convert_state_linears[js[j]](nx_i_temp))
                else:
                    nx_i_ls.append(nx_i_temp)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            
        return F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))) + \
               self.fc_m_layers[i](m_i)

class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h*n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               F.relu(self.fc_m_layers[i](m_i)) + a_i

class LToSMultiAgentPolicy(Policy):
    """Multi-agent policy that implements Learning to Share (LToS) approach with high and low level policies"""
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, n_step, shared_dim=64, n_fc=64, use_lstm=True, identical=True, agent_id=0):
        # Store the full lists of dimensions
        self.n_s_ls = n_s_ls if isinstance(n_s_ls, list) else [n_s_ls]
        self.n_a_ls = n_a_ls if isinstance(n_a_ls, list) else [n_a_ls]
        self.n_neighbors = int(torch.sum(neighbor_mask[agent_id]))
        
        # Get this agent's dimensions (first element)
        n_s = self.n_s_ls[agent_id]
        n_a = self.n_a_ls[agent_id]
        
        super().__init__(n_a, n_s, n_step, 'lstm', 'lstm', identical)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_s = n_s  # This agent's state dimension
        self.n_a = n_a  # This agent's action dimension
        self.neighbor_mask = neighbor_mask
        
        # print(f"n_neighbors: {self.n_neighbors}")
        self.n_step = n_step
        self.shared_dim = shared_dim
        self.n_fc = n_fc
        self.use_lstm = use_lstm
        self.identical = identical
        self.agent_id = agent_id
        
        # Initialize tracking attributes for tensorboard
        self.q_loss = 0.0
        self.policy_loss = 0.0
        self.total_loss = 0.0
        self.actor_loss = 0.0
        self.q_values = None
        self.w_out = None
        self.w_in = None
        self.w_in_grads = None
        self.epsilon = 1.0  # Initial exploration rate
        
        # Initialize networks
        self._init_net()
        
        # Initialize LSTM states
        self.states_fw = None
        self.target_states_fw = None
        self._reset()

    def _init_net(self):
        """Initialize neural networks."""
        # High-level policy (w_out)
        self.phi = nn.Sequential(
            nn.Linear(self.n_s, self.n_fc),
            nn.ReLU(),
            nn.Linear(self.n_fc, self.shared_dim)
        )
        
        # Target high-level policy
        self.target_phi = nn.Sequential(
            nn.Linear(self.n_s, self.n_fc),
            nn.ReLU(),
            nn.Linear(self.n_fc, self.shared_dim)
        )
        
        # Low-level policy
        if self.use_lstm:
            self.lstm_cell = nn.LSTMCell(
                self.n_s + self.shared_dim * self.n_neighbors,  # input size: obs + neighbor embeddings
                self.shared_dim
            )
            self.actor_head = nn.Linear(self.shared_dim, self.n_a)
            
            # Target networks for LSTM
            self.target_lstm_cell = nn.LSTMCell(
                self.n_s + self.shared_dim * self.n_neighbors,
                self.shared_dim
            )
            self.target_actor_head = nn.Linear(self.shared_dim, self.n_a)
        else:
            self.actor_layer = nn.Sequential(
                nn.Linear(self.n_s + self.shared_dim * self.n_neighbors, self.n_fc),
                nn.ReLU(),
                nn.Linear(self.n_fc, self.n_a)
            )
            self.target_actor_layer = nn.Sequential(
                nn.Linear(self.n_s + self.shared_dim * self.n_neighbors, self.n_fc),
                nn.ReLU(),
                nn.Linear(self.n_fc, self.n_a)
            )
        
        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(self.shared_dim + self.n_a, self.n_fc),
            nn.ReLU(),
            nn.Linear(self.n_fc, 1)
        )
        self.target_q_net = nn.Sequential(
            nn.Linear(self.shared_dim + self.n_a, self.n_fc),
            nn.ReLU(),
            nn.Linear(self.n_fc, 1)
        )
        
        # Initialize target networks
        self.target_phi.load_state_dict(self.phi.state_dict())
        if self.use_lstm:
            self.target_lstm_cell.load_state_dict(self.lstm_cell.state_dict())
            self.target_actor_head.load_state_dict(self.actor_head.state_dict())
        else:
            self.target_actor_layer.load_state_dict(self.actor_layer.state_dict())
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def compute_w_out(self, obs, agent_id, epsilon=0.1):
        """Compute high-level policy outputs (shared embeddings) for a single agent with exploration noise"""
        # Convert observation to tensor
        obs_i = torch.from_numpy(np.array(obs)).float().to(self.device)
            
        # Add batch dimension if needed
        if obs_i.dim() == 1:
            obs_i = obs_i.unsqueeze(0)
            
        w = self.phi(obs_i)
        if random.random() < epsilon:
            # Add noise for exploration
            w = w + torch.randn_like(w) * 0.1
            
        # Store w_out for tensorboard
        self.w_out = w.detach()
            
        return w 

    def compute_critic(self, w_vec, a):
        """Compute Q-values for given embeddings and actions
        
        Args:
            w_vec: Weight vector (embedding)
            a: Action (can be None for value estimation)
        """
        if a is None:
            # For value estimation, compute Q-values for all possible actions
            a = torch.arange(self.n_a, device=self.device)
            w_vec = w_vec.expand(self.n_a, -1)  # Expand for all actions
            # Create one-hot encoding for all actions
            a = F.one_hot(a, self.n_a).float()  # Shape: [n_a, n_a]
        else:
            # Convert action to one-hot encoding
            a = F.one_hot(a.long(), self.n_a).float()
            if a.dim() == 1:
                a = a.unsqueeze(0)  # Add batch dimension
            if w_vec.dim() == 1:
                w_vec = w_vec.unsqueeze(0)  # Add batch dimension
            # Expand w_vec to match action dimensions
            w_vec = w_vec.expand(a.size(0), -1)
        
        # Concatenate weight vector and action
        input_vec = torch.cat([w_vec, a], dim=-1)
        
        # Pass through Q-network
        q_values = self.q_net(input_vec)
        
        # Store Q-values for tensorboard
        self.q_values = q_values.detach()
        
        return q_values

    def compute_target_critic(self, w_vec, a):
        """Compute target Q-values for given embeddings and actions
        
        Args:
            w_vec: Weight vector (embedding)
            a: Action (can be None for value estimation)
        """
        if a is None:
            # For value estimation, compute Q-values for all possible actions
            a = torch.arange(self.n_a, device=self.device)
            w_vec = w_vec.unsqueeze(0).expand(self.n_a, -1)  # Expand for all actions
            a = F.one_hot(a, self.n_a).float()
        else:
            # Convert action to one-hot encoding
            a = F.one_hot(a.long(), self.n_a).float()
            if a.dim() == 1:
                a = a.unsqueeze(0)  # Add batch dimension
            if w_vec.dim() == 1:
                w_vec = w_vec.unsqueeze(0)  # Add batch dimension
            w_vec = w_vec.unsqueeze(1).expand(a.size(0), a.size(1), -1)  # Expand to [120,6,32]
        
        # Concatenate weight vector and action
        input_vec = torch.cat([w_vec, a], dim=-1)
        
        # Pass through target Q-network
        q_values = self.target_q_net(input_vec)
        
        return torch.max(q_values, dim=1)[0].squeeze(-1)  # Max over actions (dim=1), then squeeze last dim to get [120]

    def compute_reward_sharing(self, rewards, weights=None):
        """Compute shared rewards between neighboring agents with optional weights"""
        rewards = torch.tensor(rewards).float().to(self.device)
        r_shared = []
        
        for i in range(len(rewards)):
            neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
            if len(neighbors) == 0:
                r_shared.append(rewards[i])
                continue
                
            if weights is not None:
                # Use provided weights
                neighbor_weights = weights[i][neighbors]
            else:
                # Uniform weights
                neighbor_weights = torch.ones(len(neighbors), device=self.device) / len(neighbors)
            
            # Vectorized computation
            shared = torch.sum(neighbor_weights * rewards[neighbors])
            r_shared.append(shared)
            
        return torch.stack(r_shared)

    def compute_gradients(self, obs, w_in, actions):
        '''This function computes the gradient of the Q-value with respect to the input weights (w_in) for each agent.'''
        """compute $g_i^{\mathrm{in}}=\nabla_{w_i^{\mathrm{in}}} q_i^{\pi_i}\left(o_i, \arg \max _{a_i} q_i^{\pi_i} ; w_i^{\mathrm{in}}\right)$"""
        # Get number of neighbors
        n_neighbors = w_in.size(1)  # 4 neighbors
        
        # Initialize list to store gradients for each neighbor
        w_in_grads = []
        
        # Compute gradient for each neighbor's input weights
        for n in range(n_neighbors):
            # Get current neighbor's weights
            w_in_n = w_in[:, n, :]  # [120, 32]
            
            # First compute Q-values for all possible actions
            a_all = torch.arange(self.n_a, device=self.device)  # [n_a]
            a_all_1hot = F.one_hot(a_all, self.n_a).float()  # [n_a, n_a]
            
            # Expand w_in_n for all actions
            w_in_n_expanded = w_in_n.unsqueeze(1).expand(-1, self.n_a, -1)  # [120, n_a, 32]
            a_all_1hot_expanded = a_all_1hot.unsqueeze(0).expand(w_in_n.size(0), -1, -1)  # [120, n_a, n_a]
            
            # Reshape for Q-network
            w_in_n_flat = w_in_n_expanded.reshape(-1, w_in_n.size(-1))  # [120*n_a, 32]
            a_all_1hot_flat = a_all_1hot_expanded.reshape(-1, self.n_a)  # [120*n_a, n_a]
            
            # Compute Q-values for all actions
            q_input = torch.cat([w_in_n_flat, a_all_1hot_flat], dim=-1)  # [120*n_a, 32 + n_a]
            q_values = self.q_net(q_input)  # [120*n_a, 1]
            q_values = q_values.reshape(w_in_n.size(0), self.n_a)  # [120, n_a]
            
            # Get argmax actions
            max_actions = q_values.argmax(dim=1)  # [120]
            max_actions_1hot = F.one_hot(max_actions, self.n_a).float()  # [120, n_a]
            
            # Compute Q-values for max actions
            q_input_max = torch.cat([w_in_n, max_actions_1hot], dim=-1)  # [120, 32 + n_a]
            q_values_max = self.q_net(q_input_max)  # [120, 1]
            
            # Compute gradients with error handling
            try:
                w_in_grad = torch.autograd.grad(
                    q_values_max.sum(),  # Sum to get scalar for gradient
                    w_in_n,
                    create_graph=True,
                    retain_graph=True
                )[0]
            except RuntimeError:
                # Handle case where gradient computation fails
                w_in_grad = torch.zeros_like(w_in_n)
                
            w_in_grads.append(w_in_grad)
            
        # Stack gradients for all neighbors
        w_in_grads = torch.stack(w_in_grads, dim=1)  # [120, 4, 32]
        
        # Store gradients for tensorboard
        self.w_in_grads = w_in_grads.detach()
        
        return w_in_grads

    def compute_actions(self, obs, w_in, done=None, epsilon=0.1):
        """Compute low-level policy actions using observations and neighbor embeddings"""
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_i = torch.from_numpy(obs).float().to(self.device)
        else:
            obs_i = obs
        
        # Ensure w_in is a list of tensors
        if not isinstance(w_in, list):
            w_in = [w_in]
        
        # Store w_in for tensorboard
        self.w_in = w_in
        
        # Get number of neighbors for this agent
        n_n = len(w_in) # 4 
        # print(f"n_n: {n_n}")
        
        if n_n > 0:  # Agent has neighbors
            # Concatenate neighbor weights
            w_in_cat = torch.cat(w_in, dim=-1)
            
            # Ensure both tensors have the same number of dimensions
            if obs_i.dim() == 1:
                obs_i = obs_i.unsqueeze(0)  # Add batch dimension  # [1, 48]
            if w_in_cat.dim() == 1:
                w_in_cat = w_in_cat.unsqueeze(0)  # Add batch dimension [1, 128] 4*32
            
            # Concatenate observation and weights
            input_vec = torch.cat([obs_i, w_in_cat], dim=-1)
        else:  # Agent has no neighbors
            # Just use the observation
            if obs_i.dim() == 1:
                obs_i = obs_i.unsqueeze(0)  # Add batch dimension
            input_vec = obs_i
        
        if self.use_lstm:
            h, c = torch.chunk(self.states_fw, 2, dim=1)
            if done is not None:
                # Reset LSTM states if episode is done
                done = torch.tensor(done, dtype=torch.float32).to(self.device)
                h = h * (1 - done)
                c = c * (1 - done)
            
            # Expand hidden states to match batch size
            batch_size = input_vec.size(0)
            h = h.expand(batch_size, -1)  # [batch_size, hidden_dim]
            c = c.expand(batch_size, -1)  # [batch_size, hidden_dim]
            
            h_new, c_new = self.lstm_cell(input_vec, (h, c))
            self.states_fw = torch.cat([h_new, c_new], dim=1).detach()
            logits = self.actor_head(h_new)
        else:
            logits = self.actor_layer(input_vec)
        
        logits = F.softmax(logits, dim=1).squeeze()
        
        return logits.detach().cpu().numpy() # return logits for each action

    def _reset(self):
        """Reset LSTM states."""
        if self.use_lstm:
            self.states_fw = torch.zeros(1, self.shared_dim * 2, device=self.device)
            self.target_states_fw = torch.zeros(1, self.shared_dim * 2, device=self.device)

    def soft_update(self, tau=0.01):
        """Soft update target networks using polyak averaging"""
        # Update target networks
        for target_param, param in zip(self.target_phi.parameters(), self.phi.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        if self.use_lstm:
            for target_param, param in zip(self.target_lstm_cell.parameters(), self.lstm_cell.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.target_actor_head.parameters(), self.actor_head.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            for target_param, param in zip(self.target_actor_layer.parameters(), self.actor_layer.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def soft_update_target_phi(self, tau=0.01):
        """Soft update target high-level policy (θ_i')"""
        for target_param, param in zip(self.target_phi.parameters(), self.phi.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def soft_update_target_policy(self, tau=0.01):
        """Soft update target low-level policy (μ_i')"""
        if self.use_lstm:
            for target_param, param in zip(self.target_lstm_cell.parameters(), self.lstm_cell.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.target_actor_head.parameters(), self.actor_head.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            for target_param, param in zip(self.target_actor_layer.parameters(), self.actor_layer.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # Also update target Q-network
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def soft_update(self, tau=0.01):
        """Legacy method that updates all target networks at once"""
        self.soft_update_target_phi(tau)
        self.soft_update_target_policy(tau)

    def _update_tensorboard(self, summary_writer, global_step):
        """Update tensorboard with LToS-specific metrics.
        
        Args:
            summary_writer: Tensorboard summary writer
            global_step: Current training step
        """
        if global_step is None:
            return
            
        # Track high-level policy (θ_i) metrics
        if hasattr(self, 'q_loss'):
            summary_writer.add_scalar(f'agent_{self.agent_id}/high_level/q_loss', self.q_loss, global_step)
        if hasattr(self, 'policy_loss'):
            summary_writer.add_scalar(f'agent_{self.agent_id}/high_level/policy_loss', self.policy_loss, global_step)
        if hasattr(self, 'total_loss'):
            summary_writer.add_scalar(f'agent_{self.agent_id}/high_level/total_loss', self.total_loss, global_step)
        
        # Track low-level policy (μ_i) metrics
        if hasattr(self, 'actor_loss'):
            summary_writer.add_scalar(f'agent_{self.agent_id}/low_level/actor_loss', self.actor_loss, global_step)
        
        # Track Q-value statistics
        if hasattr(self, 'q_values') and self.q_values is not None:
            summary_writer.add_scalar(f'agent_{self.agent_id}/q_values/mean', torch.mean(self.q_values), global_step)
            summary_writer.add_scalar(f'agent_{self.agent_id}/q_values/max', torch.max(self.q_values), global_step)
            summary_writer.add_scalar(f'agent_{self.agent_id}/q_values/min', torch.min(self.q_values), global_step)
        
        # Track weight statistics
        if hasattr(self, 'w_out') and self.w_out is not None:
            summary_writer.add_scalar(f'agent_{self.agent_id}/weights/w_out_norm', torch.norm(self.w_out), global_step)
        if hasattr(self, 'w_in') and self.w_in is not None:
            if isinstance(self.w_in, list) and len(self.w_in) > 0:
                w_in_norm = torch.mean(torch.stack([torch.norm(w) for w in self.w_in]))
                summary_writer.add_scalar(f'agent_{self.agent_id}/weights/w_in_norm', w_in_norm, global_step)
        
        # Track gradient statistics
        if hasattr(self, 'w_in_grads') and self.w_in_grads is not None:
            grad_norm = torch.norm(self.w_in_grads)
            summary_writer.add_scalar(f'agent_{self.agent_id}/gradients/w_in_grad_norm', grad_norm, global_step)
        
        # Track exploration
        if hasattr(self, 'epsilon'):
            summary_writer.add_scalar(f'agent_{self.agent_id}/exploration/epsilon', self.epsilon, global_step)
