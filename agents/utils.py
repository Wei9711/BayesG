import numpy as np
import torch
import torch.nn as nn

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)


def run_rnn(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)


def one_hot(x, oh_dim, dim=-1):
    device = x.device  # Get the device of input tensor
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape,  device=device)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma # Discount factor used to discount future rewards.
        self.alpha = alpha # A scaling factor applied to spatial rewards, typically used to adjust the influence of rewards based on the distance.
        if alpha > 0:
            # A mask used for spatial reward computations. 
            #   It determines how rewards should be adjusted based on the relative distance between agents or other elements in the environment.
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []   #  List of observations.
        self.acts = []  #  List of actions taken by the agent.
        self.rs = []    #  List of rewards received by the agent
        self.vs = []    #  List of value estimates for each state
        self.adds = []  #  List of agent IDs or neighborhood information (for multi-agent settings).
        self.dones = [done] # List of done flags (whether the episode is finished after each step)

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R) # Computes the return and advantage using standard reward-to-go.
        else:
            self._add_s_R_Adv(R) # Computes the return and advantage using spatial reward adjustments.
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    '''Computes the return and advantage using standard reward-to-go.'''
    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r.cpu().numpy() if isinstance(r, torch.Tensor) else r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    '''Computes the return and advantage with spatial reward adjustments, using a distance-based mask'''
    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = torch.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    '''Computes the return and advantage using spatial reward adjustments'''
    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            # additional spatial rewards
            for t in range(self.max_distance + 1):
                if isinstance(r, np.ndarray):
                    r = torch.from_numpy(r).float()
                rt = torch.sum(r[self.distance_mask == t])
                R += (self.alpha ** t) * rt
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)

    def sample_transition(self, R, dt=0):
        '''
        Inputs:
            R:          Final reward signal.
            dt:         Temporal difference, relevant for distance-based rewards.

        Returns: 
            obs:        Observations.
            policies:   Policy logits or distributions.
            acts:       Actions taken.
            dones:      Done flags.
            Rs:         Discounted returns.
            Advs:       Advantages.
        '''
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        
        ''' Transposes and converts stored trajectories (obs, policies, acts) into NumPy arrays for further processing. '''    
        obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        acts = np.transpose(np.array(self.acts, dtype=np.int32))
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=bool)
        
        '''Resets the buffer, retaining only the last done value.'''
        self.reset(self.dones[-1])

        return obs, policies, acts, dones, Rs, Advs

    '''Computes the discounted returns (Rs) and advantages (Advs) for each agent (Standard reward-to-go.)'''
    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        '''Loops through each agent's rewards (self.rs), value estimates (self.vs), and done flags (self.dones)'''
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = r + self.gamma * cur_R * (1.-done)
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    '''
    Extends _add_R_Adv by incorporating spatial rewards based on:
        distance_mask:  Defines which rewards are relevant at different distances.
        max_distance:   Maximum distance for reward propagation.
    '''
    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                '''Adds a spatial reward correction term by iterating over distances up to max_distance'''
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    if isinstance(r, np.ndarray):
                        r = torch.from_numpy(r).float()
                    rt = torch.sum(r[distance_mask == t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                
                cur_R = self.gamma * cur_R * (1.-done)
                # additional spatial rewards
                for t in range(max_distance + 1):
                    if isinstance(r, np.ndarray):
                        r = torch.from_numpy(r).float()
                    rt = torch.sum(r[distance_mask == t])
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)


class LToSPolicyBuffer(OnPolicyBuffer):
    """Buffer for storing transitions in LToS algorithm, including input weights."""
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)
        self.w_in = []  # Store input weights from neighbors

    def reset(self, done=False):
        # explicitly reset required fields without clearing w_in
        self.obs = []
        self.adds = []
        self.acts = []
        self.rs = []
        self.Rs = []
        self.Advs = []
        self.vs = []
        self.dones = [done]  # maintain initial done flag
        # Do NOT reset self.w_in here if you intend to keep its history

    def add_transition(self, ob, na, a, r, v, done, w_in=None):
        if w_in is not None:
            # w_in is a list of tensors; clone and detach each tensor individually
            # cloned_w_in = [w.clone().detach() for w in w_in]
            self.w_in.append(w_in)
        super().add_transition(ob, na, a, r, v, done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)  # Computes the return and advantage using standard reward-to-go.
        else:
            self._add_s_R_Adv(R)  # Computes the return and advantage using spatial reward adjustments.
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=np.bool)
        vs = np.array(self.vs, dtype=np.float32)
        # Get the corresponding w_in for the sampled transitions
        stored_w_in = self.w_in[:len(obs)]  # Only return w_in for the current batch

         # Clear w_in after sampling
        self.w_in = self.w_in[len(obs):]  # Keep only the remaining w_in

        self.reset(self.dones[-1])  # Reset after sampling
        return obs, nas, acts, dones, Rs, Advs, stored_w_in

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val

