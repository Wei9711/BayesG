#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2021 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
from collections import deque

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
import sys
import optparse
import random
import sumolib
import numpy as np
import gym
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
from copy import deepcopy as dp


import json


class Large_city_net(gym.Wrapper):
    def __init__(self, net_path,sim_path,folder_name=None,reward_scale=10):
        self.name = "Large_city"
        self.obj = "queue"     # queue   wait  arrived
        self.folder_name = folder_name
        self.net_path = net_path
        self.sim_path = sim_path
        self.net = sumolib.net.readNet(self.net_path)
        self.AdjacencyList = self.generateTopology()    #获取邻接矩阵
        print("self.AdjacencyList=",self.AdjacencyList)
        self.Edges = self.net.getEdges()
        self.VEH_LEN_M = 200
        self.coop_gamma = -1
        self.T = 500
        # self.max_steps = self.T
        # self.cur_step = 0
        self.reward_scale = reward_scale


        
        
        self.gui = False
        if self.gui:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.traci = traci
        self.traci.start([self.sumoBinary, "-c", sim_path,"--no-warnings"])



        self.n_agent = self.traci.trafficlight.getIDCount()
        self.n_agents = self.n_agent
        if self.n_agent < 500:
            self.single_step_second = 20
        else:
            self.single_step_second = 40
        print("n_agent= ", self.n_agent)     #交通灯的数量
        self.id_list = list(self.traci.trafficlight.getIDList())
        print("id_list= ", self.id_list)       #交通灯 的id
        #phase = traci.trafficlight.getAllProgramLogics(id_list[119])
        # print("phase=",len(phase[0].phases))    #action 的数量
        
        self.A  = []
        for i in range(self.n_agent):
            phase = self.traci.trafficlight.getAllProgramLogics(self.id_list[i])
            a = len(phase[0].phases)
            self.A.append(a)
            
        self.n_action = max(self.A)
        self.action_space = Discrete(self.n_action)
        
        # print("A=",max(A))       ##action最多的交通灯 = 动作空间的维度
        self.neighbor_mask = self.get_neighbor_matrix_khop(k=1)
        # Print graph statistics
        adj = self.neighbor_mask
        num_nodes = adj.shape[0]
        # Remove self-loops for edge counting
        adj_no_self = adj.copy()
        np.fill_diagonal(adj_no_self, 1)
        num_edges = int(np.sum(adj_no_self))
        degrees = np.sum(adj_no_self, axis=1)
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        print(f"Graph stats: nodes={num_nodes}, edges={num_edges}, avg_degree={avg_degree:.2f}, max_degree={max_degree}")
        # print("self.neighbor_mask.shape=",self.neighbor_mask.shape)
        print("self.neighbor_mask=",self.neighbor_mask.sum(1))
        # print("self.neighbor_mask_khop2=",self.neighbor_mask_khop2.sum(1))
        # print("self.neighbor_mask_edges=",self.neighbor_mask_edges.sum(1))
        # print("high order matrix k=2:",self.get_neighbor_matrix_khop(k=2).sum(1))
        # print("high order matrix k=1:",self.get_neighbor_matrix_khop(k=1).sum(1))
        print("Total TLS:", len(self.id_list))
        print("Connected TLS nodes:")
        # for i in range(len(self.id_list)):
        #     num_neighbors = np.sum(self.neighbor_mask[i]) - 1  # exclude self
            # if num_neighbors == 0:
            #     print(f"TLS {self.id_list[i]} has NO neighbors")

        # self.write_neighbor_graph_poly()
        self.distance_mask = dp(self.neighbor_mask)
        self.E_id, self.E_from, self.E_to, self.E_to_state, self.E_from_state = self.get_edge_matrix()

        self.node_names = self.id_list
        self.ILDS_in = []
        self.CAP=[]
        ss = []
        for node_name in self.node_names:
            lanes_in = self.traci.trafficlight.getControlledLanes(node_name)
            ilds_in = []
            lanes_cap = []
            for lane_name in lanes_in:
                cur_ilds_in = [lane_name]
                ilds_in.append(cur_ilds_in)
                cur_cap = 0
                for ild_name in cur_ilds_in:
                    cur_cap += self.traci.lane.getLength(ild_name)
                lanes_cap.append(cur_cap/float(self.VEH_LEN_M))
            ss.append(len(lanes_cap))
            self.ILDS_in.append(ilds_in)
            self.CAP.append(lanes_cap)
        self.n_s = max(ss)
        print("ss=",ss)
        self.n_s_ls = [self.n_s]*self.n_agent
        self.n_a_ls = [self.n_action]*self.n_agent
        print("state_space=",self.n_s)
        self.fp = np.ones((self.n_agent, self.n_action)) / self.n_action

        self.state_heterogeneous_space = ss
        for i in range(self.n_agent):
            self.state_heterogeneous_space[i] = ss[i]
            indices = np.where(self.neighbor_mask[i] == 1)[0]
            for j in range(len(indices)):
                self.state_heterogeneous_space[i]+=ss[j]
        # np.savetxt('state_heterogeneous_space.csv', np.array(self.state_heterogeneous_space), delimiter=',')


        self.traci.close()
        sys.stdout.flush()

    def cal_n_order_matrix(self,n_nodes,max_order,adj):
        def calculate_high_order_adj(max_order):
            result_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
            for i in range(n_nodes):
                for j in range(n_nodes):                  
                    if abs(j-i) <= max_order:
                        result_matrix[i][j] = 1 

            return result_matrix

        adjacency_matrix = np.eye(n_nodes)
        result = calculate_high_order_adj(max_order)
        for q in range(n_nodes):
            for k in range(n_nodes):
                if adj[q][k]==1:
                    result[q][k]=1
        return result - np.eye(n_nodes)



    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction
    
    def get_neighbor_matrix(self):
        self.neighbor_mask = np.eye(self.n_agent)
        for i in range(self.n_agent):
            if self.id_list[i] in self.AdjacencyList:
                L = list(self.AdjacencyList[self.id_list[i]].keys())
                l1 = len(L)
                for j in range(l1):
                    if L[j] in self.id_list:
                        index = self.id_list.index(L[j])
                        self.neighbor_mask[i][index] = 1
        return self.neighbor_mask
    

    #     return neighbor_mask
    def get_neighbor_matrix_tls_khop(self, k=3):
        tls_ids = traci.trafficlight.getIDList()

        # Map each TLS ID to a single representative junction node
        tls_to_node = {}
        for tls_id in tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            if lanes:
                # Take the first lane as representative
                edge_id = traci.lane.getEdgeID(lanes[0])
                edge = self.net.getEdge(edge_id)
                node_id = edge.getToNode().getID()
                tls_to_node[tls_id] = node_id

        tl_node_ids = list(tls_to_node.values())
        id_map = {tl_id: idx for idx, tl_id in enumerate(tl_node_ids)}
        n = len(tl_node_ids)
        print("n=", n)
        neighbor_mask = np.eye(n)

        for i, from_node in enumerate(tl_node_ids):
            queue = deque([(from_node, 0)])
            visited = {from_node: 0}

            while queue:
                current_node, depth = queue.popleft()

                if depth >= k:
                    continue

                try:
                    outgoing_edges = self.net.getNode(current_node).getOutgoing()
                except KeyError:
                    continue  # Skip if node ID is missing

                for edge in outgoing_edges:
                    next_node = edge.getToNode().getID()

                    if next_node not in visited or visited[next_node] > depth + 1:
                        visited[next_node] = depth + 1

                        if next_node in id_map and next_node != from_node and visited[next_node] <= k:
                            neighbor_mask[i][id_map[next_node]] = 1

                        queue.append((next_node, depth + 1))

        return neighbor_mask






    def get_neighbor_matrix_khop_2(self, k=2):
        tls_list = self.net.getTrafficLights()
        tl_node_ids = [tls.getID() for tls in tls_list]
        id_map = {tl_id: idx for idx, tl_id in enumerate(tl_node_ids)}
        n = len(tl_node_ids)
        neighbor_mask = np.eye(n)

        for i, from_node in enumerate(tl_node_ids):
            visited = {from_node: 0}
            queue = deque([(from_node, 0)])

            while queue:
                current_node, depth = queue.popleft()
                if depth >= k:
                    continue

                try:
                    outgoing_edges = self.net.getNode(current_node).getOutgoing()
                except KeyError:
                    continue

                for edge in outgoing_edges:
                    to_node = edge.getToNode().getID()

                    # Only increment the depth for nodes not visited yet or visited with a higher depth
                    if to_node not in visited or visited[to_node] > depth + 1:
                        visited[to_node] = depth + 1
                        queue.append((to_node, depth + 1))

                        # Add edge only if it's exactly within k hops and it's a traffic light node
                        if 0 < visited[to_node] <= k and to_node in id_map and to_node != from_node:
                            j = id_map[to_node]
                            neighbor_mask[i][j] = 1

        return neighbor_mask



    def get_neighbor_matrix_khop(self, k=2):
        """
        Builds a k-hop neighbor matrix among traffic light-controlled junctions.
        Handles cases where traffic light IDs do not directly map to node IDs.
        """
        tls_list = self.net.getTrafficLights()  # sumolib.TLS objects
        tl_node_ids = [tls.getID() for tls in tls_list]  # true junction node IDs

        id_map = {tl_id: idx for idx, tl_id in enumerate(tl_node_ids)}
        n = len(tl_node_ids)
        neighbor_mask = np.eye(n)

        for i, from_node in enumerate(tl_node_ids):
            visited = set()
            queue = deque([(from_node, 0)])
            visited.add(from_node)

            while queue:
                current_node, depth = queue.popleft()
                if depth >= k:
                    continue

                try:
                    outgoing_edges = self.net.getNode(current_node).getOutgoing()
                except KeyError:
                    continue  # Skip if node not found in net (shouldn't happen now)

                for edge in outgoing_edges:
                    to_node = edge.getToNode().getID()

                    if to_node in visited:
                        continue
                    visited.add(to_node)

                    if to_node in id_map and to_node != from_node:
                        j = id_map[to_node]
                        neighbor_mask[i][j] = 1

                    queue.append((to_node, depth + 1))
        
        return neighbor_mask


    def get_edge_matrix(self):
        E_id = []
        E_from = []
        E_to = []
        
        for i in range(len(self.Edges)):
            E_id.append(self.Edges[i].getID())
            E_from.append(self.Edges[i].getFromNode().getID())
            E_to.append(self.Edges[i].getToNode().getID())
        
        E_to_state = [] 
        E_from_state = [] 
        for p in range(self.n_agent):
            E_index_to = [index for index, value in enumerate(E_to) if value == self.id_list[p]]
            E_index_from = [index for index, value in enumerate(E_from) if value == self.id_list[p]]
            e_to = []
            e_from = []
            for q in E_index_to:
                e_to.append(E_id[q])
            for w in E_index_from:
                e_from.append(E_id[w])
            E_to_state.append(e_to)
            E_from_state.append(e_from)
        return E_id, E_from, E_to, E_to_state,E_from_state
    

    def generateTopology(self):	
        AdjacencyList = {}
        for e in self.net.getEdges():
            if AdjacencyList.__contains__(str(e.getFromNode().getID()))==False:
                AdjacencyList[str(e.getFromNode().getID())]={}
            AdjacencyList[str(e.getFromNode().getID())][str(e.getToNode().getID())] = e.getLanes()[0].getLength()
        return AdjacencyList
    
    def get_tls_positions(self):
        """
        Get average coordinates for each traffic light using the to-node of each controlled lane's edge.
        This version avoids getToNode() on Lane directly, which doesn't exist.
        """
        positions = {}
        tl_ids = traci.trafficlight.getIDList()

        for tl_id in tl_ids:
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                coords = []
                for lane_id in controlled_lanes:
                    try:
                        edge = self.net.getLane(lane_id).getEdge()
                        to_node = edge.getToNode()
                        coords.append(to_node.getCoord())
                    except Exception as e:
                        print(f"[Lane lookup error] {lane_id}: {e}")
                if coords:
                    x_avg = sum(p[0] for p in coords) / len(coords)
                    y_avg = sum(p[1] for p in coords) / len(coords)
                    positions[tl_id] = (x_avg, y_avg)
                else:
                    print(f"[Warning] No coords found for TL {tl_id}")
            except Exception as e:
                print(f"[TL error] {tl_id}: {e}")
        return positions


    
    def write_neighbor_graph_poly(self, xml_name="neighbor_graph.add.xml"):
        neighbor_mask = self.neighbor_mask
        positions = self.get_tls_positions()
        print("positions=",len(positions))  
        
        with open(f"tls_positions.json", "w") as f:
            json.dump(positions, f)

        tls_ids = [tls.getID() for tls in self.net.getTrafficLights()]
        print("tls_ids=",len(tls_ids))
        print("neighbor_mask.shape=",neighbor_mask.shape)   
        assert neighbor_mask.shape[0] == len(tls_ids), "Mismatch between matrix and tls_ids length"

        filename = f"{xml_name}"
        with open(filename, "w") as f:
            f.write('<additional>\n')
            for i in range(len(tls_ids)):
                for j in range(len(tls_ids)):
                    if neighbor_mask[i][j] == 1 and i != j:
                        if tls_ids[i] not in positions or tls_ids[j] not in positions:
                            continue  # skip missing TLs
                        x1, y1 = positions[tls_ids[i]]
                        x2, y2 = positions[tls_ids[j]]
                        shape = f"{x1},{y1} {x2},{y2}"
                        poly_id = f"edge_{i}_{j}"
                        f.write(f'    <polyline id="{poly_id}" color="red" shape="{shape}" layer="2"/>\n')
            f.write('</additional>\n')

    # def get_options(self):
    #     optParser = optparse.OptionParser()
    #     optParser.add_option("--nogui", action="store_true",
    #                          default=True, help="run the commandline version of sumo")
    #     options, args = optParser.parse_args()
    #     return options

    # def get_state(self,old_state):
    #     cur_state_delta = []
    #     #print("old_state=",state) 
    #     for k in range(self.n_agent): 
    #         cur_wave_to = 0    
    #         cur_wave_from = 0       
    #         for o in range(len(self.E_to_state[k])):
    #             cur_wave_to += self.traci.edge.getLastStepVehicleNumber(self.E_to_state[k][o]) 
    #         #print("cur_wave=",cur_wave)
    #         cur_state_delta.append(cur_wave_to)
    #     cur_state_delta = np.array(cur_state_delta)
    #     # print("cur_state=",cur_state)             #状态信息，获取每个路口上个时刻驶入的车辆数量
    #     new_state = old_state + cur_state_delta
    #     # print("new_state=",new_state) 

    #     return new_state



    def get_state(self):
        cur_state = []
        for k, ild in enumerate(self.ILDS_in):

            cur_wave = []
            for j, ild_seg in enumerate(ild):
                cur_wave.append(self.traci.lane.getLastStepVehicleNumber(ild_seg[0])/self.CAP[k][j])
            cur_state.append(cur_wave)
        # cur_state = np.array(cur_state)

        #Pad each sublist to length self.n_s
        cur_state_padded = np.array([
            np.pad(sublist, (0, self.n_s - len(sublist)), constant_values=0.0)
            for sublist in cur_state
        ])
        # cur_state_padded = np.array([np.pad(sublist, (0, self.n_s - len(sublist)), constant_values=0.0) for sublist in cur_state])

        return cur_state_padded



    def get_reward(self):

        if self.obj == "queue":

            queues = []
            for k, ild in enumerate(self.ILDS_in):
                cur_queue = 0
                for j, ild_seg in enumerate(ild):
                    cur_queue += self.traci.lane.getLastStepHaltingNumber(ild_seg[0])
                queues.append(cur_queue)
            reward = -np.array(queues)/float(self.reward_scale)


        elif self.obj == "wait":
            waits = []
            for k, ild in enumerate(self.ILDS_in):
                for j, ild_seg in enumerate(ild):
                    max_pos = 0
                    cur_cars = self.traci.lane.getLastStepVehicleIDs(ild_seg[0])
                    for vid in cur_cars:
                        car_pos = self.traci.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.traci.vehicle.getWaitingTime(vid)   
                            waits.append(car_wait)
            reward = -np.array(waits)/float(self.reward_scale)



        return reward


        
    def reset(self):

        # if self.gui:
        #     self.sumoBinary = checkBinary('sumo-gui')
        # else:
        #     self.sumoBinary = checkBinary('sumo')
        # self.cur_step = 0  # reset step counter
        self.traci.start([self.sumoBinary, "-c", self.sim_path,"--no-warnings"])


        self.state_heterogeneous_space


        #state = np.zeros((self.n_agent, self.n_s))

        state = []
        for i in range(len(self.state_heterogeneous_space)):
            state.append(np.zeros((1,self.state_heterogeneous_space[i]))[0])

        state = np.zeros((self.n_agent, self.n_s))
        self.state = state


        return self.state 

    def clear(self):
        self.traci.close()
        
        # Safely flush stdout
        try:
            sys.stdout.flush()
        except Exception as e:
            print(f"Warning: Could not flush stdout: {e}")
            pass
        return
       
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):

        #self.traci.trafficlight.setRedYellowGreenState(node_name, phase)
        #self.traci.trafficlight.setPhaseDuration(node_name, phase_duration)
        # self.cur_step += 1

        for i in range(self.n_agent):
            if action[i] <= self.A[i]-1:
                self.traci.trafficlight.setPhase(self.id_list[i], action[i])   
            else:
                a = action[i]
                while a > self.A[i]-1:
                    a = a - self.A[i]
                self.traci.trafficlight.setPhase(self.id_list[i], a)   

        self.arrived = []
        for _ in range(self.single_step_second):   
            self.traci.simulationStep()


            if self.obj == "arrived":
                arrived_vehicles = self.traci.simulation.getArrivedIDList()
                num_arrived_vehicles = len(arrived_vehicles)
                if num_arrived_vehicles>0:
                    self.arrived.append(num_arrived_vehicles)
        
        
        state_old = self.state
        state = self.get_state()
        
        
        if self.obj == "arrived":
            reward = np.repeat(sum(self.arrived), self.n_agent)
        else:
            reward = self.get_reward()

        

        reward = np.array(reward, dtype=np.float32)
        done = False  # Initialize done as False
        self.state = state

        reward = np.sum(reward)
        return state, reward, done, reward

    def get_state_(self):
        state = self.state
        return state


def Large_city_Env(net_path,sim_path,folder_name,reward_scale):
    # print("parent_dir=",parent_dir) /home/wduan/Data
    # net_path = parent_dir + net_path   # 436 agents
    # sim_path = parent_dir + sim_path
    return Large_city_net(net_path,sim_path,folder_name,reward_scale)



# this is the main entry point of this script
if __name__ == "__main__":
    current_dir = "./"
    net_path = current_dir + "NewYork167/newyork167.net.xml"    # 436 agents
    sim_path = current_dir + "NewYork167/newyork167.sumocfg"
    folder_name = None
    env = Large_city_Env(net_path,sim_path,folder_name,10)
    env.reset()



    # for i in range(100):
    #     action = [5]*59
    #     state, reward, done, info = env.step(action)
    #     print("state=",state.shape)
    #     # print("reward=",reward)

