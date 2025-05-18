import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
import shutil
import datetime


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


# def copy_file(src_dir, tar_dir):
#     cmd = ' cp %s %s' % (src_dir, tar_dir)
#     subprocess.check_call(cmd, shell=True)
def copy_file(src_dir, tar_dir):
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(tar_dir), exist_ok=True)
        shutil.copy(src_dir, tar_dir)
        print(f"File copied successfully from {src_dir} to {tar_dir}")
    except Exception as e:
        print(f"Error while copying file: {e}")

def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model'], custom_log_dir=None, config=None):
    """Initialize directories for logging, data, and models.
    
    Args:
        base_dir: Base directory path
        pathes: List of subdirectories to create
        custom_log_dir: Custom log directory path if provided
        config: Configuration object containing model settings
    """
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    
    for path in pathes:
        if path == 'log' and custom_log_dir:
            # Use custom log directory if provided
            cur_dir = custom_log_dir
            
            # If using GAT, add number of heads to log path
            if config and 'MODEL_CONFIG' in config:
                is_graph_nn = config['MODEL_CONFIG'].getboolean('is_graph_nn', False)
                if is_graph_nn:
                    gnn_type = config['MODEL_CONFIG'].get('gnn_type', 'gat')
                    if gnn_type == 'gat':
                        n_heads = config['MODEL_CONFIG'].getint('n_attention_heads', 4)
                        cur_dir = os.path.join(cur_dir, f'{gnn_type}_heads_{n_heads}')
                    elif gnn_type == 'gcn' or gnn_type == 'sage':
                        cur_dir = os.path.join(cur_dir, f'{gnn_type}')
                    else:
                        raise ValueError(f'Invalid GNN type: {gnn_type}')
                    is_mlp_gnn = config['MODEL_CONFIG'].getboolean('is_mlp_gnn', False)
                    if is_mlp_gnn:
                        base_dir = os.path.basename(cur_dir)  # Get the last part of the path (xxx)
                        parent_dir = os.path.dirname(cur_dir)  # Get the parent directory path
                        cur_dir = os.path.join(parent_dir, f'{base_dir}_mlp_gnn')  # Construct the new path

        else:
            cur_dir = base_dir + '/%s/' % path

        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")    
        cur_dir   = os.path.join(cur_dir, timestamp)
        # Create all necessary parent directories
        os.makedirs(cur_dir, exist_ok=True)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop

'''
Trainer class is responsible for managing the training and evaluation processes of RL setup. 
It interacts with the environment, the model, and the global counters while logging and saving results.
'''
class Trainer():
    def __init__(self, env, env_name,  algo_name, model, global_counter, summary_writer, output_path=None, logger=None):
        '''
            env:                The environment object where agents operate.
            model:              The reinforcement learning model.
            global_counter:     A shared counter to track global training steps.
            summary_writer:     Used for logging training summaries (e.g., TensorBoard).
            output_path:        Directory to save training logs and results.
        '''
        self.cur_step = 0                       # Tracks the steps taken in the current episode.
        self.global_counter = global_counter    # 
        self.env = env
        self.env_name = env_name
        self.algo_name =algo_name
        if self.env_name == "Pandemic" or self.env_name == "Large_city":
            if self.algo_name == "ConseNet":
                self.agent = "ma2c_cu"
            elif self.algo_name == "IA2C_FP":
                self.agent = "ia2c_fp"
            elif self.algo_name == "IA2C":
                self.agent = "ia2c"
            elif self.algo_name == "IA2C_LToS":
                self.agent = "ia2c_ltos"
            else:
                self.agent = "ma2c_cu"
        else:
            self.agent = self.env.agent
                       # Gets the agent type from the environment (e.g., single or multi-agent).
        self.model = model
        if self.env_name == "slowdown" or self.env_name == "catchup":
            self.n_step = 600
        elif self.env_name == "Grid" or self.env_name == "Monaco":
            self.n_step = 720 # self.n_step = self.model.n_step
        elif self.env_name == "PowerGrid":
            self.n_step = self.env.T
        elif self.env_name == "Pandemic":
            self.n_step = self.env.T
        elif self.env_name == "Large_city":
            self.n_step = self.env.T
        else:       
            self.n_step = self.model.n_step
                    # The step size for multi-step returns, derived from the model.
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []                          # A list to log rewards and statistics for each episode.
        self.output_path = output_path
        self.env.train_mode = True              # Sets the environment to training mode.
        self.logger = logger
        
    '''Logs rewards to the summary writer (e.g., TensorBoard).'''
    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    '''
    Computes the agent's policy given the current observation and whether the episode is done. 
    The policy can be stochastic (for training) or deterministic (for testing).
    '''
    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            # get the fingerprint of the policies of neighboring agents
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            # Check if pi contains any NaN values
            if np.any(np.isnan(pi)):
                # Replace NaN values with -1000000 to make those actions have near-zero probability
                pi = np.nan_to_num(pi, nan=-1000000)
                
                # Convert logits to probabilities using softmax
                pi = np.exp(pi - np.max(pi))  # Subtract max for numerical stability
                pi = pi / np.sum(pi)  # Normalize to sum to 1
            
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    '''Retrieves the value estimation of the current state and action'''
    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    '''Handles the exploration phase during training'''
    def explore(self, prev_ob, prev_done):
        ob = prev_ob

        if self.env_name == "Large_city":
            self.env.clear()
            self.env.reset()
            ob = self.env.get_state_()

        done = prev_done

        self.ep_reward = 0

        # Loop through the number of steps in the episode
        '''Collect experience for n_step steps'''
        for _ in range(self.n_step):
            # This generates the actor's policy (probability distribution over actions)
            policy, action = self._get_policy(ob, done)
            # This generates the critic's value estimate for the given action
            # action: list of 25 actions (each action is an integer) list[int]
            value = self._get_value(ob, done, action)
            # transition
            # policy: list of 25 policies (each policy is a list of 5 actions) list[array[5]]
            self.env.update_fingerprint(policy)
            # This takes the action and returns the next state, reward, and whether the episode is done
            # next_ob, reward, done, global_reward = self.env.step(action)

            if self.env_name == "slowdown" or self.env_name == "catchup":
                #print("action=",action)

                next_ob, reward, done, _ = self.env.step(action)
                
                s = dp(np.array(next_ob))
                self.S.append(s.ravel())
                
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

                #print("next_ob=",next_ob)
                #print("self.env.coop_gamma=",self.env.coop_gamma)
                #print("reward=",reward)
                #print("done=",done)
                #print("global_reward=",global_reward)

            elif self.env_name == "Grid":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

                #print("next_ob=",next_ob)
                #print("reward=",reward)
                #print("done=",done)
                #print("global_reward=",global_reward)    

            else:
                next_ob, reward, done, global_reward = self.env.step(action)
            

            episode_r = reward
            if episode_r.ndim > 1:
                episode_r = episode_r.mean(axis=0)
            
            #print("episode_r=",episode_r)
            self.ep_reward += episode_r

            '''... store experience ...'''
            # Append the global reward to the episode rewards list  
            self.episode_rewards.append(global_reward) #self.episode_rewards.append(global_reward/self.env.n_agents)

            # Increment the global step counter
            global_step = self.global_counter.next()
            # Increment the current step counter
            self.cur_step += 1
            # collect experience
            if self.agent.startswith('ma2c'):
                # self.ps refers to the "fingerprint" of the policies of neighboring agents. I
                self.model.add_transition(ob, self.ps, action, reward, value, done) 
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                # logging.info('''Training: global step %d, episode step %d,
                #                    ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                #              (global_step, self.cur_step,
                #               str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
                logging.info('''Training: global step %d, episode step %d, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step, global_reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            if done:
                if self.env_name != "catchup":
                    break
                else:
                    continue
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent)
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)

        if self.env_name == "Grid":
            episode_reward = np.array(self.ep_reward).sum()
            logging.info(f"Episode reward: {episode_reward}")
            self.summary_writer.add_scalar('episode_reward', episode_reward, global_step=self.global_counter.cur_step)
        elif self.env_name == "Monaco":
            episode_reward = global_reward
            logging.info(f"Episode reward: {episode_reward}")
        elif self.env_name == "Large_city":
            episode_reward = np.array(self.ep_reward).sum()
            logging.info(f"Episode reward: {episode_reward}")
            self.summary_writer.add_scalar('episode_reward', episode_reward, global_step=self.global_counter.cur_step)

        return ob, done, R

    '''
    Evaluates the agent's performance on a test environment. The test policy is typically stochastic for on-policy methods.
    '''
    def perform(self, test_ind, gui=False):
        if self.env_name == "Pandemic":
            self.env.reset()
            ob = self.env.get_state_()
        elif self.env_name == "Large_city":
            self.env.clear()
            self.env.reset()
            ob = self.env.get_state_()
        else:
            ob = self.env.reset(gui=gui, test_ind=test_ind)

        rewards = []
        done = True
        self.model.reset()
        step = 0
        
        # Get control interval from the environment's simulator
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'control_interval_sec'):
            control_interval = self.env.sim.control_interval_sec
            logging.info(f"Found control_interval_sec in simulator: {control_interval}")
        else:
            # Try to get it from the environment's config
            if hasattr(self.env, 'config') and hasattr(self.env.config, 'getint'):
                try:
                    control_interval = self.env.config.getint('control_interval_sec')
                    logging.info(f"Found control_interval_sec in config: {control_interval}")
                except:
                    control_interval = 1
                    logging.warning("Could not find control_interval_sec in config, defaulting to 1")
            else:
                control_interval = 1
                logging.warning("Could not find control_interval_sec, defaulting to 1")
            
        logging.info(f"Starting evaluation with control_interval: {control_interval}")
        
        while True:
            print(f"step: {step}")
            print(f"control_interval: {control_interval}")
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
                
                # Use simulation time for mask visualization triggers
                current_time = getattr(self.env, 'cur_sec', step)
                if current_time % (500 * control_interval) == 0 and hasattr(self.model, 'visualize_masks'):
                    logging.info(f"Visualizing masks at simulation time {current_time}")
                    if self.mask_viz_dir is not None:
                        try:
                            self.model.visualize_masks(current_time, save_path=self.mask_viz_dir)
                            logging.info(f"Successfully saved mask visualization to {self.mask_viz_dir}")
                        except Exception as e:
                            logging.error(f"Error saving mask visualization: {str(e)}")
                    else:
                        try:
                            self.model.visualize_masks(current_time)
                            logging.info("Successfully displayed mask visualization")
                        except Exception as e:
                            logging.error(f"Error displaying mask visualization: {str(e)}")

            if self.env_name == "slowdown" or self.env_name == "catchup":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Grid" or self.env_name == "Monaco":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Pandemic":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Large_city":
                next_ob, reward, done, global_reward = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            else:
                next_ob, reward, done, global_reward = self.env.step(action)

            rewards.append(global_reward)
            # No need to increment step manually; rely on env.cur_sec
            if done:
                logging.info(f"Episode finished at step {step}")
                break
            ob = next_ob
            
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    '''The core training loop'''
    def run(self):
        self.S = []
        ii = 0
        while not self.global_counter.should_stop():
            ''''1. Resets the environment and model at the start of each episode.'''
            # np.random.seed(self.env.seed)
            if self.env_name == "Pandemic":
                self.env.reset()
                ob = self.env.get_state_()
            elif self.env_name == "Large_city":
                if ii != 0:
                    self.env.clear()
                self.env.reset()
                ob = self.env.get_state_()
            else:
                ob = self.env.reset()
            ii+=1
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            '''2. Calls the explore method to interact with the environment.'''
            while True:
                ob, done, R = self.explore(ob, done)
                # R is the return of the episode | len 25 list[float]
                # dt is the number of steps remaining in the episode
                dt = self.env.T - self.cur_step 
                global_step = self.global_counter.cur_step

                if self.env_name == "Pandemic":
                    R = [float(x) for x in R]
                    
                '''Uses the collected data to train the model'''
                self.model.backward(R, dt, self.summary_writer, global_step)
                # termination
                if done:
                    # try:
                        # Try terminate() first for standard environments
                    self.env.terminate()
                    # except (AttributeError, RecursionError):
                    #     # If terminate() doesn't exist or causes recursion, try clear()
                    #     try:
                    #         self.env.clear()
                    #     except AttributeError:
                    #         # If neither method exists, log a warning
                    #         logging.warning("Environment has neither terminate() nor clear() method")
                    # pytorch implementation is faster, wait SUMO for 1s
                    time.sleep(1)
                    break
                if global_step >= self.global_counter.total_step:
                    break
            '''3. Computes rewards and logs the episode results'''
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            '''4. Periodically tests the model's performance.'''
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc') and not self.env.name.startswith('Large_city'):
                self.env.train_mode = False
                '''Periodically evaluates the model by calling perform'''
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, env_name, model, output_path, gui=False):
        self.env = env
        self.env_name = env_name
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.gui = gui
        print(f"Model Name : {self.model.name}")
        
        # Create a default output directory if none provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            self.output_path = os.path.join(os.getcwd(), 'demo_output', timestamp)
        else:
            self.output_path = output_path
            
        # Create directory for mask visualizations
        self.mask_viz_dir = os.path.join(self.output_path, 'mask_visualizations')
        os.makedirs(self.mask_viz_dir, exist_ok=True)
        logging.info(f'Saving mask visualizations to: {self.mask_viz_dir}')

    def perform(self, test_ind, gui=False):
        if self.env_name == "Pandemic":
            self.env.reset()
            ob = self.env.get_state_()
        elif self.env_name == "Large_city":
            self.env.clear()
            self.env.reset()
            ob = self.env.get_state_()
        else:
            ob = self.env.reset(gui=gui, test_ind=test_ind)

        rewards = []
        done = True
        self.model.reset()
        step = 0
        
        # Get control interval from the environment's simulator
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'control_interval_sec'):
            control_interval = self.env.sim.control_interval_sec
            logging.info(f"Found control_interval_sec in simulator: {control_interval}")
        else:
            # Try to get it from the environment's config
            if hasattr(self.env, 'config') and hasattr(self.env.config, 'getint'):
                try:
                    control_interval = self.env.config.getint('control_interval_sec')
                    logging.info(f"Found control_interval_sec in config: {control_interval}")
                except:
                    control_interval = 1
                    logging.warning("Could not find control_interval_sec in config, defaulting to 1")
            else:
                control_interval = 1
                logging.warning("Could not find control_interval_sec, defaulting to 1")
            
        logging.info(f"Starting evaluation with control_interval: {control_interval}")
        
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
                
                # Use simulation time for mask visualization triggers
                current_time = getattr(self.env, 'cur_sec', step)
                print(f"current_time: {current_time}")
                print(f"Has visualize_masks method: {hasattr(self.model, 'visualize_masks')}")
                print(f"Time check: {current_time % (500 * control_interval)}")
                if current_time % (200 * control_interval) == 0 and hasattr(self.model, 'visualize_masks'):
                    logging.info(f"Visualizing masks at simulation time {current_time}")
                    if self.mask_viz_dir is not None:
                        try:
                            self.model.visualize_masks(current_time, save_path=self.mask_viz_dir, draw_whole=True)
                            logging.info(f"Successfully saved mask visualization to {self.mask_viz_dir}")
                        except Exception as e:
                            logging.error(f"Error saving mask visualization: {str(e)}")
                    else:
                        try:
                            self.model.visualize_masks(current_time)
                            logging.info("Successfully displayed mask visualization")
                        except Exception as e:
                            logging.error(f"Error displaying mask visualization: {str(e)}")

            if self.env_name == "slowdown" or self.env_name == "catchup":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Grid" or self.env_name == "Monaco":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Pandemic":
                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Large_city":
                next_ob, reward, done, global_reward = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            else:
                next_ob, reward, done, global_reward = self.env.step(action)

            rewards.append(global_reward)
            # No need to increment step manually; rely on env.cur_sec
            if done:
                logging.info(f"Episode finished at step {step}")
                break
            ob = next_ob
            
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
