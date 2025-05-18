

import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler, LToSPolicyBuffer
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, 
                             NCMultiAgentPolicy, 
                             CommNetMultiAgentPolicy, 
                             DIALMultiAgentPolicy, 
                             GraphCMultiAgentPolicy, 
                             BayesianGraphCMultiAgentPolicy,
                             NCMultiAgentPolicy_MLP,
                             LToSMultiAgentPolicy)
import logging
import numpy as np
import torch.nn.functional as F

'''
Advantage Actor-Critic (IA2C) algorithm:
Decentralized actor and centralized critic approach, limited to interactions within a neighborhood.
'''
class IA2C:
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c'
        '''
        Initializes the IA2C algorithm with parameters like:
            state/action dimensions:            (n_s_ls, n_a_ls)
            neighborhood and distance masks:    (neighbor_mask, distance_mask)
            Hyperparameters:                     discount factor (coop_gamma),  reward_clip, reward_norm, learning rate scheduler
        '''
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, naction, action, reward, value, done):
        # Convert inputs to tensors if they aren't already
        reward = torch.as_tensor(reward, device=self.device)
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(ob[i], naction[i], action[i], 
                                              reward, value[i], done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if i == 0:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    #forward() is used for both action selection (actor) and value estimation (critic)
    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent): 
            cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    def load(self, model_path, train_mode=False):
        # If model_path is a file, load directly
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # If model_path is a directory, use the old logic
        elif os.path.isdir(model_path):
            # Check if current policy is GraphCMultiAgentPolicy
            if isinstance(self.policy, GraphCMultiAgentPolicy):
                graph_model_path = os.path.join(model_path, 'graph_model.pt')
                if os.path.exists(graph_model_path):
                    checkpoint = torch.load(graph_model_path, map_location=torch.device('cpu'))
                else:
                    if train_mode:
                        print('No existing GraphCMultiAgentPolicy model, starting training from scratch')
                        return False
                    else:
                        raise ValueError('GraphCMultiAgentPolicy model not found')
            else:
                checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=torch.device('cpu'))
        else:
            raise ValueError(f"Provided model_path {model_path} is neither a file nor a directory.")

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        if train_mode:
            self.policy.train()
        else:
            self.policy.eval()
        return True

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        '''
        n_s_ls:         A list of state dimensions for each agent.
        n_a_ls:         A list of action dimensions for each agent.
        neighbor_mask:  A mask defining the connectivity or neighborhood relations among agents (likely a graph adjacency matrix).
        distance_mask:  A mask related to distance between agents (possibly used for computing proximity-related interactions).
        coop_gamma:     A cooperation discount factor.
        total_step:     Total number of steps for training (used for initializing the training process).
        seed:           A random seed to ensure reproducibility.
        use_gpu:        A boolean flag indicating whether GPU should be used.
        model_config:   A configuration object (likely a dictionary or config parser) storing hyperparameters and other settings.

        '''
        # Store model_config as instance variable
        self.model_config = model_config
        
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        
        
        '''Agent Identification'''
        self.identical_agent = False
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        
        '''Neighborhood Setup'''
        self.n_agent = len(neighbor_mask)

        '''Reward Normalization and Clipping'''
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')

        # Number of steps in an episode . (How many sequential steps of experience to collect before doing a learning update. 
        #                                  This is different from the traditional supervised learning definition of batch size (which refers to the number of independent samples processed together).)
        self.n_step = model_config.getint('batch_size') # 120
        
        '''Neural Network Layers'''
        self.n_fc = model_config.getint('num_fc') # 64
        self.n_lstm = model_config.getint('num_lstm') # 64

        '''Device Configuration'''
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('Use gpu for pytorch...')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('Use cpu for pytorch...')
        
        # Convert masks to tensors and move to device
        self.neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.long, device=self.device)
        self.distance_mask = torch.tensor(distance_mask, dtype=torch.float, device=self.device)

        '''Policy Initialization'''
        self.policy = self._init_policy()
        # Move entire policy to device at once instead of per-layer
        self.policy = self.policy.to(self.device)
        
        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)

    '''
    The method creates a separate policy network for each agent,
        taking into account whether the agents are identical or heterogeneous.
    '''
    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            # Calculate the Number of Neighbors (n_n)
            n_n = int(torch.sum(self.neighbor_mask[i]))
            if self.identical_agent:
                '''
                    Input dimension (self.n_s_ls[i]).
                    Action dimension (self.n_a_ls[i]).
                    Number of neighbors (n_n).
                    Time steps for LSTM (self.n_step).
                    Fully connected and LSTM layers (self.n_fc and self.n_lstm).
                '''
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                '''Heterogeneous Agents'''
                na_dim_ls = [] # A list of action dimensions for the neighbors of agent i
                for j in torch.where(self.neighbor_mask[i] == 1)[0]:
                        na_dim_ls.append(self.n_a_ls[j])
                '''
                    Neighbor action dimensions (na_dim_ls).
                '''    
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i),
                                          na_dim_ls=na_dim_ls, identical=False)
                # local_policy.to(self.device)
            policy.append(local_policy)
        return nn.ModuleList(policy)

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init, 
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(gamma, coop_gamma, distance_mask[i]))

    def _update_lr(self):
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class IA2C_FP(IA2C):
    """
    In fingerprint IA2C, neighborhood policies (fingerprints) are also included.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c_fp'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, 
                        total_step, seed, use_gpu, model_config)
    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = torch.sum(self.neighbor_mask[i])
            # neighborhood policies are included in local state
            if self.identical_agent:
                n_s1 = int(self.n_s_ls[i] + self.n_a*n_n) 
                policy.append(FPPolicy(n_s1, self.n_a, int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i)))
            else:
                na_dim_ls = []
                for j in torch.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                n_s1 = int(self.n_s_ls[i] + sum(na_dim_ls))
                policy.append(FPPolicy(n_s1, self.n_a_ls[i], int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i),
                                       na_dim_ls=na_dim_ls, identical=False))
        return nn.ModuleList(policy)


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0,  use_gpu=False, gnn_type='gat'):
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, p, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.identical_agent:
            self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                             reward, value, done)
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    # backward() is used for updating the policy and value networks  
    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)

        self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                             summary_writer=summary_writer, global_step=global_step)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    #forward() is used for both action selection ('p' actor) and value estimation ('v' critic)
    def forward(self, obs, done, ps, actions=None, out_type='p'):
        if self.identical_agent:
            return self.policy.forward(np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p

    # def _init_policy(self):
    #     if self.identical_agent:
    #         return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
    #                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
    #     else:
    #         return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
    #                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
    #                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)
    def _init_policy(self):
        # Add is_graph_nn to model_config with default False
        print(f"Config is_graph_nn value: {self.model_config.getboolean('is_graph_nn', False)}")
        self.is_graph_nn = self.model_config.getboolean('is_graph_nn', False)
        self.gnn_version = self.model_config.get('gnn_version', 'v2')  # Default to 'v2' if not set
        
        # Common parameters for both policy types
        policy_params = {
            'n_s': self.n_s, # number of states
            'n_a': self.n_a, # number of actions
            'n_agent': self.n_agent, # number of agents
            'n_step': self.n_step, # number of steps
            'neighbor_mask': self.neighbor_mask, # neighbor mask
            'n_fc': self.n_fc, # number of features
            'n_h': self.n_lstm # number of hidden layers
        }
        
        # Add parameters for heterogeneous agents
        if not self.identical_agent:
            policy_params.update({
                'n_s_ls': self.n_s_ls,
                'n_a_ls': self.n_a_ls,
                'identical': False
            })
        
        # Choose policy based on is_graph_nn flag
        if self.is_graph_nn:
            # Add GAT-specific parameters
            policy_params['gnn_type'] = self.model_config.get('gnn_type', 'gcn')
            policy_params['n_heads'] = self.model_config.getint('n_attention_heads', 4)  # default 4 heads
            policy_params['unify_act_state_dim'] = self.model_config.getboolean('unify_act_state_dim', False)
            policy_params['use_random_mask'] = self.model_config.getboolean('use_random_mask', False)
            return GraphCMultiAgentPolicy(**policy_params)
        else:
            self.unify_act_state_dim = self.model_config.getboolean('unify_act_state_dim', False)
            if not self.identical_agent and self.unify_act_state_dim:
                policy_params['unify_act_state_dim'] = self.unify_act_state_dim
                return NCMultiAgentPolicy_MLP(**policy_params)
            else:
                return NCMultiAgentPolicy(**policy_params)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)

class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
        self.policy.consensus_update()

class BayesianGraph(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'bayesian_graph'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def visualize_masks(self, step, save_path=None, draw_whole=False):
        """Delegate visualization to the policy's visualize_masks method"""
        if hasattr(self.policy, 'visualize_masks'):
            self.policy.visualize_masks(step, save_path, draw_whole)
        else:
            logging.warning("Policy does not have visualize_masks method")

    def _init_policy(self):
        self.is_graph_nn = self.model_config.getboolean('is_graph_nn', False)
        self.unify_act_state_dim = self.model_config.getboolean('unify_act_state_dim', False)
        if self.is_graph_nn:
            # Add GAT-specific parameters
            self.gnn_type = self.model_config.get('gnn_type', 'gcn')
            self.heads = self.model_config.getint('n_attention_heads', 4)  # default 4 heads
            self.learn_mask = self.model_config.getboolean('learn_mask', True)
            return BayesianGraphCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                                  self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                                  n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls,
                                                  identical=self.identical_agent, unify_act_state_dim=self.unify_act_state_dim,
                                                  n_heads=self.heads,gnn_type=self.gnn_type, learn_mask=self.learn_mask)
        
        else:
            raise ValueError("Bayesian Graph is not supported for non-graph models")

class MA2C_DIAL(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_dial'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

class MA2C_CNET(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_ic3'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                           n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

class IA2C_LToS(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        # Initialize shared_dim before calling parent's __init__
        self.shared_dim = model_config.getint('shared_dim', fallback=64)
        # Ensure n_fc matches shared_dim to maintain consistent dimensions
        self.n_fc = self.shared_dim  # Set n_fc equal to shared_dim
        self.use_lstm = model_config.getboolean('use_lstm', fallback=True)
        self.tau = model_config.getfloat('tau', fallback=0.01)
        self.grad_clip = model_config.getfloat('grad_clip', fallback=10.0)
        self.update_frequency = model_config.getint('update_frequency', fallback=1)
        self.gradient_steps = model_config.getint('gradient_steps', fallback=1)
        self.gamma = model_config.getfloat('gamma', fallback=0.99)
        
        # Initialize epsilon parameters
        self.epsilon = model_config.getfloat('epsilon', fallback=1.0)
        self.epsilon_min = model_config.getfloat('epsilon_min', fallback=0.01)
        self.epsilon_decay = model_config.getfloat('epsilon_decay', fallback=0.995)
        self.current_epsilon = self.epsilon
        
        # Initialize loss tracking
        self.q_losses = [0.0] * len(n_s_ls)
        self.policy_losses = [0.0] * len(n_s_ls)
        
        # Initialize step counter
        self.step_count = 0
        
        # Call parent's __init__ after initializing our attributes
        super().__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, model_config, seed, use_gpu)
        
        self.w_in = [[] for _ in range(self.n_agent)]  # Store input weights
        self.w_out = [[] for _ in range(self.n_agent)]  # Store output weights

    def _init_policy(self):
        """Initialize LToS policy networks for each agent."""
        policies = []
        for i in range(self.n_agent):
            # print(f"Initializing LToS policy for agent {i}")
            policy = LToSMultiAgentPolicy(
                n_s_ls=self.n_s_ls,  
                n_a_ls=self.n_a_ls,  
                neighbor_mask=self.neighbor_mask,
                n_step=self.n_step,
                shared_dim=self.shared_dim,
                n_fc=self.n_fc,
                use_lstm=self.use_lstm,
                identical=self.identical_agent,
                agent_id=i
            ).to(self.device)
            policies.append(policy)
        return nn.ModuleList(policies)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        """Initialize transition buffers for each agent.
        
        Args:
            gamma: Discount factor
            distance_mask: Mask for distance-based reward sharing
            coop_gamma: Cooperation coefficient
        """
        self.trans_buffer = []
        for i in range(self.n_agent):
            self.trans_buffer.append(LToSPolicyBuffer(
                gamma, coop_gamma, distance_mask[i]))

    def add_transition(self, ob, naction, action, reward, value, done):
        # Handle reward normalization and clipping in numpy
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
            
        # Store transitions with w_in
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(
                ob[i], naction[i], action[i], reward, value[i], done, self.w_in[i])

    def forward(self, obs, done, naction=None, out_type='p'):
        """Forward pass to compute actions or values
        
        Args:
            obs: List of observations for each agent
            done: Done flag
            out_type: 'p' for policy (actions) or 'v' for value estimation
        """
        # Reset states if done
        if done:
            self.reset()
            
        # Compute output weights (high-level policy)
        self.w_out = []
        for i in range(self.n_agent):
            w = self.policy[i].compute_w_out(obs[i], agent_id=i, epsilon=self.current_epsilon)
            self.w_out.append(w)
        
        # Compute input weights from neighbors
        self.w_in = [[] for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
            for j in neighbors:
                self.w_in[i].append(self.w_out[j])
        
        if out_type == 'p':
            # Return actions
            actions = []
            for i in range(self.n_agent):
                # Ensure states are properly sized
                if self.use_lstm:
                    self.policy[i].states_fw = self.policy[i].states_fw[:1]  # Keep only first state
                    self.policy[i].target_states_fw = self.policy[i].target_states_fw[:1]  # Keep only first state
                
                action = self.policy[i].compute_actions(obs[i], self.w_in[i], done=done, epsilon=self.current_epsilon)
                actions.append(action)
            return actions
        else:  # out_type == 'v'
            # Return value estimates
            values = []
            for i in range(self.n_agent):
                # Get Q-values for each action
                q_values = self.policy[i].compute_critic(self.w_out[i], None)  # Pass None for action to get all Q-values
                # Take max Q-value as value estimate
                value = torch.max(q_values).item()
                values.append(value)
            return values

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.step_count += 1
        
        # Only update if it's time to update
        if self.step_count % self.update_frequency != 0:
            return

        for _ in range(self.gradient_steps):
            self.optimizer.zero_grad()
            
            for i in range(self.n_agent):
                # Sample transitions from replay buffer
                obs, nactions, actions, dones, Rs, Advs, stored_w_in = self.trans_buffer[i].sample_transition(Rends[i], dt)
                
                # Skip if no observations
                if len(obs) == 0:
                    continue
                    
                batch_size = len(obs)  # |D| - size of the minibatch
                obs = torch.from_numpy(obs).float().to(self.device)
                actions = torch.from_numpy(actions).long().to(self.device)
                dones = torch.from_numpy(dones).float().to(self.device)
                Rs = torch.from_numpy(Rs).float().to(self.device)
                Advs = torch.from_numpy(Advs).float().to(self.device)
                
                # Convert stored_w_in from list of lists of tensors to a single tensor
                # Filter out empty lists and stack in one pass
                non_empty_w_in = [w_list for w_list in stored_w_in if w_list]
                if non_empty_w_in:
                    stored_w_in = torch.stack([torch.stack(w_list) for w_list in non_empty_w_in]).to(self.device)
                    stored_w_in = stored_w_in.squeeze(2)  # [batch_size, n_neighbors, w_dim]
                else:
                    # If no stored weights, create zero tensor with expected shape
                    n_neighbors = int(torch.sum(self.neighbor_mask[i]))
                    stored_w_in = torch.zeros((batch_size, n_neighbors, self.shared_dim), device=self.device)

                # Compute w_out for current observations - process entire batch at once
                w_out = self.policy[i].phi[0](obs)  # [720, 32]

                # Compute target Q-values
                with torch.no_grad():
                    # Get neighbors' target weights
                    neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
                    n_neighbors = int(torch.sum(self.neighbor_mask[i]))
                    
                    # Skip neighbor processing if agent has no neighbors
                    if n_neighbors == 0:
                        # Create zero tensor with correct shape for no neighbors
                        next_w_in = torch.zeros((obs.size(0), 0), device=self.device)  # [120, 0]
                        next_actions = self.policy[i].compute_actions(obs, next_w_in, epsilon=0.0)
                        continue
                        
                    # Process neighbors in a single loop
                    valid_next_w_in = []  # List to store valid tensors
                    for j in neighbors:
                        # Get observations for neighbor j
                        neighbor_obs, _, _, _, _, _, _ = self.trans_buffer[j].sample_transition(Rends[j], dt)
                        
                        # Skip if no observations
                        if len(neighbor_obs) == 0:
                            continue
                            
                        neighbor_obs = torch.from_numpy(neighbor_obs).float().to(self.device)
                        next_w_out = self.policy[j].target_phi[0](neighbor_obs)  # [720, 32]
                        
                        # Ensure all weights have the same size
                        # if next_w_out.size(1) != self.shared_dim:
                        #     next_w_out = self.policy[i].phi[0](neighbor_obs)  # Use current agent's phi to ensure consistent size
                            
                        valid_next_w_in.append(next_w_out)
                    
                    # If we have fewer neighbors than expected, pad with zeros
                    if len(valid_next_w_in) < n_neighbors:
                        padding = torch.zeros((obs.size(0), self.shared_dim), device=self.device)
                        for _ in range(n_neighbors - len(valid_next_w_in)):
                            valid_next_w_in.append(padding)
                    
                    try:
                        # Stack and reshape in one operation
                        next_w_in = torch.stack(valid_next_w_in, dim=1)  # [720, n_neighbors, 32]
                        next_w_in = next_w_in.reshape(next_w_in.shape[0], -1)  # [720, n_neighbors*32]
                    except RuntimeError as e:
                        print(f"Error processing next_w_in for agent {i}:")
                        print(f"Number of neighbors in mask: {n_neighbors}")
                        print(f"Number of valid neighbors: {len(valid_next_w_in)}")
                        print(f"Observation shape: {obs.shape}")
                        if valid_next_w_in:
                            print(f"First weight shape: {valid_next_w_in[0].shape}")
                        raise e
                    
                    next_actions = self.policy[i].compute_actions(obs, next_w_in, epsilon=0.0)
                    # Convert next_actions to tensor if it's numpy
                    if isinstance(next_actions, np.ndarray):
                        next_actions = torch.from_numpy(next_actions).long().to(self.device)
                    
                    # Compute target Q-values for entire batch at once
                    target_q_vals = self.policy[i].compute_target_critic(next_w_out, next_actions)
                    y = Rs + self.gamma * target_q_vals * (1 - dones)
                
                # Compute Q-values for current state-action pairs
                q_vals = self.policy[i].compute_critic(w_out, actions).squeeze(-1)
                
                # Update μ_i (low-level policy) by minimizing MSE loss
                q_loss = F.mse_loss(q_vals, y.detach())  # Already includes 1/|D| averaging
                self.q_losses[i] = q_loss.item()
                
                # Compute policy loss with explicit batch averaging
                policy_loss = -(Advs * q_vals).mean()  # Already includes 1/|D| averaging
                self.policy_losses[i] = policy_loss.item()
                
                # Update μ_i (low-level policy) parameters
                total_loss = q_loss + policy_loss
                total_loss.backward()
                
                # Compute gradient for w_in and update high-level policy
                # obs tensor [120, 48]
                # stored_w_in tensor [120, 4, 32]
                # actions tensor [120]
                """
                compute 
                $g_i^{in}=\nabla_{w_i^{\mathrm{in}}} q_i^{\pi_i}\left(o_i, \arg \max _{a_i} q_i^{\pi_i} ; w_i^{\mathrm{in}}\right)$"""
                w_in_grads = self.policy[i].compute_gradients(obs, stored_w_in, actions)
                # w_in_grads tensor [120, 4, 32]
                
                # Get neighbors for this agent
                neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
                
                # Exchange gradients and compute g_i^out
                # First compute φ_i(o_i) for all observations in batch
                w_out = self.policy[i].phi[0](obs)  # [120, 32]
                
                # Initialize gradient accumulator for θ_i
                theta_grad = None
                
                # For each observation in the batch
                for t in range(batch_size):
                    # Get current observation's output weights
                    w_out_t = w_out[t]  # [32]
                    w_out_t.requires_grad_(True)
                    
                    # Compute gradients for all neighbors at once
                    neighbor_grads = []
                    for n, neighbor_idx in enumerate(neighbors):
                        g_in = w_in_grads[t, n]  # [32]
                        neighbor_grads.append(g_in)
                    
                    # Stack neighbor gradients
                    neighbor_grads = torch.stack(neighbor_grads)  # [n_neighbors, 32]
                    
                    # Compute gradient of φ_i(o_i) with respect to θ_i for all neighbors at once
                    phi_grad = torch.autograd.grad(
                        w_out_t,
                        self.policy[i].phi[0].parameters(),
                        grad_outputs=neighbor_grads.sum(dim=0),  # Sum gradients from all neighbors
                        create_graph=True,
                        retain_graph=True
                    )
                    
                    # Initialize or accumulate gradients
                    if theta_grad is None:
                        theta_grad = phi_grad
                    else:
                        theta_grad = [g1 + g2 for g1, g2 in zip(theta_grad, phi_grad)]
                
                # Average gradients over batch size
                theta_grad = [g / batch_size for g in theta_grad]
                
                # Update θ_i using accumulated gradients
                for param, grad in zip(self.policy[i].phi[0].parameters(), theta_grad):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad
                
                # Update parameters
                self.optimizer.step()
                
                # Update target networks with explicit separation
                # Update target high-level policy (θ_i')
                self.policy[i].soft_update_target_phi()
                # Update target low-level policy (μ_i')
                self.policy[i].soft_update_target_policy()
                
                # Update learning rate
                self._update_lr()
                
                # Update tensorboard
                if summary_writer is not None:
                    self.policy[i]._update_tensorboard(summary_writer, global_step)

    def reset(self):
        super().reset()
        self.w_in = [[] for _ in range(self.n_agent)]
        self.w_out = [[] for _ in range(self.n_agent)]
        if self.use_lstm:
            for i in range(self.n_agent):
                self.policy[i].states_fw = torch.zeros(1, self.policy[i].shared_dim * 2, device=self.device)
                self.policy[i].target_states_fw = torch.zeros(1, self.policy[i].shared_dim * 2, device=self.device)

    def _update_tensorboard(self, summary_writer, global_step):
        """Update tensorboard with training metrics."""
        if global_step is None:
            return
            
        # Log epsilon value
        summary_writer.add_scalar('train/epsilon', self.current_epsilon, global_step)
        
        # # Log losses for each agent
        # for i in range(self.n_agent):
        #     summary_writer.add_scalar(f'train/agent_{i}/q_loss', self.q_losses[i], global_step)
        #     summary_writer.add_scalar(f'train/agent_{i}/policy_loss', self.policy_losses[i], global_step)
            
        # Log average losses across agents
        summary_writer.add_scalar('train/avg_q_loss', np.mean(self.q_losses), global_step)
        summary_writer.add_scalar('train/avg_policy_loss', np.mean(self.policy_losses), global_step)