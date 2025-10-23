# BayesG Networked Multi-agent RL 
This repo implements the state-of-the-art MARL algorithms for networked system control, with observability and communication of each agent limited to its neighborhood. 

Our BayesG:
* BayesG: a decentralized actor–critic framework that jointly learns context-aware interaction graphs and policies via variational Bayesian inference over ego-graphs in large-scale networked MARL.

For fair comparison, all algorithms are applied to A2C agents, classified into two groups: IA2C contains non-communicative policies which utilize neighborhood information only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)
* LearnToShare: [Yi, Y., G. Li, Y. Wang, et al. "Learning to share in networked multi-agent reinforcement learning." NIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html)

Available MA2C algorithms:
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: [Chu, T., S. Chinchali, S. Katti."Multi-agent reinforcement learning for networked system control." ICLR 2020](https://papers.nips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html)


Available NMARL scenarios:
* **ATSC Grid**: Adaptive traffic signal control in a synthetic traffic grid.
* **ATSC Monaco**: Adaptive traffic signal control in a real-world traffic network from Monaco city.
* **NewYork 33, 51 and 167**: These networks are derived from real-world Manhattan layouts. We use a 20-second control interval (increased to 40 seconds for NewYork167) and simulate 500 steps per episode. Intersections feature heterogeneous phase designs and ILD configurations. States include normalized lane-level metrics and neighborhood-aware features. These environments pose significant challenges for coordination and generalization in decentralized MARL.

## Requirements
* Python3 == 3.8.20
* [SUMO](http://sumo.dlr.de/wiki/Installing) >= 1.1.0
* requriements.txt

## Usages
First define all hyperparameters (including algorithm and DNN structure) in a config file under `[config_dir]` ([examples](./config)), and create the base directory of each experiement `[base_dir]`. For ATSC Grid, please call [`build_file.py`](./envs/large_grid_data) to generate SUMO network files before training.

1. To train a new agent, run
~~~
python3 main.py --base-dir [base_dir] train --config-dir [config_dir]
~~~
Training config/data and the trained model will be output to `[base_dir]/data` and `[base_dir]/model`, respectively.

BayesG related config:
* ATSC Grid: config/config_BayesG_grid.ini
* ATSC Monaco: config/config_BayesG_monaco.ini
* NewYorkXX: config/config_BayesG_newyorkXX.ini

For example:
~~~
python3 main.py --base-dir /BayesG train --config-dir /Bayes/config/config_BayesG_newyork51.ini
~~~

2. To access tensorboard during training, run
~~~
tensorboard --logdir=[base_dir]/log
~~~

3. To evaluate a trained agent, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seeds]
~~~
Evaluation data will be output to `[base_dir]/eva_data`. Make sure evaluation seeds are different from those used in training.    

4. To visualize the agent behavior in ATSC scenarios, run
~~~
python3 main.py --base-dir [base_dir] evaluate --checkpoint [saved_model_path] --demo
~~~
Such as:
~~~
python main.py --base-dir /BayesG evaluate --checkpoint /BayesG/model/xxxxxcheckpoint.pt --demo
~~~
It is recommended to use only one evaluation seed for the demo run. This will launch the SUMO GUI, and [`view.xml`](./envs/large_grid_data) can be applied to visualize queue length and intersectin delay in edge color and thickness. 

## Citation
If you find this work useful, please cite our paper:
~~~
@inproceedings{duan2025bayesian,
  title={Bayesian Ego-graph Inference for Networked Multi-Agent Reinforcement Learning},
  author={Duan, Wei and Lu, Jie and Xuan, Junyu},
  booktitle={Proceedings of the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025},
  url={https://openreview.net/forum?id=3qeTs05bRL}
}
~~~
