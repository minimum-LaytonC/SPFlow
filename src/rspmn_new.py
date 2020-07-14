import numpy as np
from spn.algorithms.SPMN import SPMN
from spn.algorithms.EM import EM_optimization
import metaData, readData
from spn.structure.Base import Sum, Product, Max
from spn.structure.leaves.spmnLeaves.SPMNLeaf import State
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.algorithms.SPMNHelper import get_ds_context
import pandas as pd
from copy import deepcopy
from spn.algorithms.Inference import  likelihood
from spn.algorithms.MPE import mpe
from sklearn.feature_selection import chi2
import queue, time
from datetime import datetime
import argparse
from spn.io.Graphics import plot_spn

class S_RSPMN:
    def __init__(self,
                dataset = "crossing_traffic",
                debug = False,
                debug1 = True,
                apply_em = False,
                use_chi2 = True,
                chi2_threshold = 0.005,
                likelihood_similarity_threshold = 0.00001,
                likelihood_match = True,
                deep_match = True,
                horizon = 3,
                problem_depth = 10,
                samples = 100000,
                num_vars = None
            ):
        self.dataset = dataset
        self.debug = debug
        self.debug1 = debug1
        self.apply_em = apply_em
        self.use_chi2 = use_chi2
        self.chi2_threshold = chi2_threshold
        self.likelihood_similarity_threshold = likelihood_similarity_threshold
        self.likelihood_match = likelihood_match
        self.deep_match = deep_match
        self.horizon = horizon
        self.problem_depth = problem_depth
        self.samples = samples
        self.num_vars = num_vars

        self.s1_node_to_SIDs = dict()
        self.SID_to_branch = dict()
        self.branch_to_SIDs = dict()
        self.SID_to_s2 = dict()
        self.s1_to_s2s = dict()

        self.meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(num_vars-1)+[MetaType.UTILITY]
        self.scope = [i for i in range(len(self.meta_types))]
        self.s2_count = 0
        self.s2_scope_idx = len(self.scope)
        self.spmn = None

        if dataset == "repeated_marbles":
            partialOrder = [['s1'],['draw'],['result','reward']]
            decNode=['draw']
            utilNode=['reward']
            scopeVars=['s1','draw','result','reward']
        elif dataset == "tiger":
            partialOrder = [['s1'],['observation'],['action'],['reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','observation','action','reward']
        elif dataset == "frozen_lake":
            partialOrder = [['s1'],['action'],['observation','reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','action','observation','reward']
        elif dataset == "nchain":
            partialOrder = [['s1'],['observation'],['action'],['reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','observation','action','reward']
        elif dataset == "elevators":
            decNode=[#'decision']
                    'close-door',
                    'move-current-dir',
                    'open-door-going-up',
                    'open-door-going-down',
                ]
            obs = [
                    'elevator-at-floor-0',
                    'elevator-at-floor-1',
                    'elevator-at-floor-2',
                    'person-in-elevator-going-down',
                    'elevator-dir',
                    'person-waiting-1',
                    'person-waiting-2',
                    'person-waiting-3',
                    'person-in-elevator-going-up',
                ]
            utilNode=['reward']
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
        elif dataset == "elevators_mdp":
            decNode=[
                    'close-door',
                    'move-current-dir',
                    'open-door-going-up',
                    'open-door-going-down',
                ]
            obs = [
                    'elevator-closed[$e0]',
                    'person-in-elevator-going-down[$e0]',
                    'elevator-at-floor[$e0, $f0]',
                    'elevator-at-floor[$e0, $f1]',
                    'elevator-at-floor[$e0, $f2]',
                    'person-waiting-up[$f0]',
                    'person-waiting-up[$f1]',
                    'person-waiting-up[$f2]',
                    'elevator-dir-up[$e0]',
                    'person-waiting-down[$f0]',
                    'person-waiting-down[$f1]',
                    'person-waiting-down[$f2]',
                    'person-in-elevator-going-up[$e0]',
                ]
            utilNode=['reward']
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
        elif dataset == "skill_teaching":
            decNode=[
                    'giveHint-1',
                    'giveHint-2',
                    'askProb-1',
                    'askProb-2',
                ]
            obs = [
                    'hintedRightObs-1',
                    'hintedRightObs-2',
                    'answeredRightObs-1',
                    'answeredRightObs-2',
                    'updateTurnObs-1',
                    'updateTurnObs-2',
                    'hintDelayObs-1',
                    'hintDelayObs-2',
                ]
            utilNode=['reward']
            #scopeVars=['s1']+obs+decNode+['reward']
            #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
        elif dataset == "skill_teaching_mdp":
            decNode=[
                    'giveHint-1',
                    'giveHint-2',
                    'askProb-1',
                    'askProb-2',
                ]
            obs = [
                    'hintDelayVar[$s0]',
                    'hintDelayVar[$s1]',
                    'updateTurn[$s0]',
                    'updateTurn[$s1]',
                    'answeredRight[$s0]',
                    'answeredRight[$s1]',
                    'proficiencyMed[$s0]',
                    'proficiencyMed[$s1]',
                    'proficiencyHigh[$s0]',
                    'proficiencyHigh[$s1]',
                    'hintedRight[$s0]',
                    'hintedRight[$s1]',
                ]
            utilNode=['reward']
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
        elif dataset == "crossing_traffic":
            decNode=['decision']
                #     'move-east',
                #     'move-north',
                #     'move-south',
                #     'move-west'
                # ]
            obs = [
                    'arrival-max-xpos-1',
                    'arrival-max-xpos-2',
                    'arrival-max-xpos-3',
                    'robot-at[$x1, $y1]',
                    'robot-at[$x1, $y2]',
                    'robot-at[$x1, $y3]',
                    'robot-at[$x2, $y1]',
                    'robot-at[$x2, $y2]',
                    'robot-at[$x2, $y3]',
                    'robot-at[$x3, $y1]',
                    'robot-at[$x3, $y2]',
                    'robot-at[$x3, $y3]',
                ]
            utilNode=['reward']
            #scopeVars=['s1']+obs+decNode+['reward']
            #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
            scope = [i for i in range(len(scopeVars))]
        elif dataset == "crossing_traffic_gym":
            decNode=['decision']
                #     'move-east',
                #     'move-north',
                #     'move-south',
                #     'move-west'
                # ]
            obs = [
                    'arrival-max-xpos-1',
                    'arrival-max-xpos-2',
                    'arrival-max-xpos-3',
                    'robot-at[$x1, $y1]',
                    'robot-at[$x1, $y2]',
                    'robot-at[$x1, $y3]',
                    'robot-at[$x2, $y1]',
                    'robot-at[$x2, $y2]',
                    'robot-at[$x2, $y3]',
                    'robot-at[$x3, $y1]',
                    'robot-at[$x3, $y2]',
                    'robot-at[$x3, $y3]',
                ]
            utilNode=['reward']
            #scopeVars=['s1']+obs+decNode+['reward']
            #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
            scope = [i for i in range(len(scopeVars))]
        elif dataset == "crossing_traffic_mdp":
            decNode=[
                    'move-east',
                    'move-north',
                    'move-south',
                    'move-west'
                ]
            obs = [
                    'robot-at[$x1, $y1]',
                    'robot-at[$x1, $y2]',
                    'robot-at[$x1, $y3]',
                    'robot-at[$x2, $y1]',
                    'robot-at[$x2, $y2]',
                    'robot-at[$x2, $y3]',
                    'robot-at[$x3, $y1]',
                    'robot-at[$x3, $y2]',
                    'robot-at[$x3, $y3]',
                    'obstacle-at[$x1, $y1]',
                    'obstacle-at[$x1, $y2]',
                    'obstacle-at[$x1, $y3]',
                    'obstacle-at[$x2, $y1]',
                    'obstacle-at[$x2, $y2]',
                    'obstacle-at[$x2, $y3]',
                    'obstacle-at[$x3, $y1]',
                    'obstacle-at[$x3, $y2]',
                    'obstacle-at[$x3, $y3]',
                ]
            utilNode=['reward']
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
        elif dataset == "game_of_life_mdp":
            decNode=[
                    'set[$x1, $y1]',
                    'set[$x1, $y2]',
                    'set[$x1, $y3]',
                    'set[$x2, $y1]',
                    'set[$x2, $y2]',
                    'set[$x2, $y3]',
                    'set[$x3, $y1]',
                    'set[$x3, $y2]',
                    'set[$x3, $y3]',
                ]
            obs = [
                    'alive[$x1, $y1]',
                    'alive[$x1, $y2]',
                    'alive[$x1, $y3]',
                    'alive[$x2, $y1]',
                    'alive[$x2, $y2]',
                    'alive[$x2, $y3]',
                    'alive[$x3, $y1]',
                    'alive[$x3, $y2]',
                    'alive[$x3, $y3]',
                ]
            utilNode=['reward']
            scopeVars=['s1']+obs+decNode+['reward']
            partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]

        self.decNode = decNode
        #self.obs=obs
        self.utilNode = utilNode
        self.scopeVars = scopeVars
        self.partialOrder = partialOrder
        self.dec_indices = [i for i in range(len(scopeVars)) if scopeVars[i] in decNode]
        self.util_indices = [i for i in range(len(scopeVars)) if scopeVars[i] in utilNode]

    def get_horizon_train_data(self, data, horizon):
        nans_h=np.empty(data.shape)
        nans_h[:,:,:] = np.nan
        data = np.concatenate((data,nans_h),axis=1)
        train_data_h = np.concatenate([data[:,i:self.problem_depth+i] for i in range(horizon)],axis=2)
        # add nans for s1
        nans=np.empty((train_data_h.shape[0],train_data_h.shape[1],1))
        nans[:] = np.nan
        train_data_h = np.concatenate((nans,train_data_h),axis=2)
        return train_data_h

    def get_horizon_params(self,partialOrder, decNode, utilNode, scopeVars, meta_types, horizon):
        partialOrder_h = [] + partialOrder
        for i in range(1,horizon):
            partialOrder_h += [[var+"_t+"+str(i) for var in s] for s in partialOrder[1:]]
        decNode_h = decNode+[decNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(decNode))]
        utilNode_h = utilNode+[utilNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(utilNode))]
        scopeVars_h = scopeVars + [var+"_t+"+str(i) for i in range (1,horizon) for var in scopeVars[1:]]
        meta_types_h = meta_types+meta_types[1:]*(horizon-1)
        return partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h

    def replace_nextState_with_s2(self, spmn):
        s1 = spmn.children[0]
        self.s1_to_s2s[s1] = list()
        scope_t1 = {i for i in range(self.s2_scope_idx)}
        q = queue.Queue()
        q.put(spmn)
        while not q.empty():
            node = q.get()
            if isinstance(node, Product):
                terminal = False
                to_remove = []
                for child in node.children:
                    # if the child has no variables from the first timestep
                    if len(set(child.scope) & scope_t1) == 0:
                        # then remove it to be replaced with an s2 node
                        to_remove.append(child)
                        terminal = True
                    else:
                        q.put(child)
                if terminal:
                    for child in to_remove:
                        node.children.remove(child)
                    new_s2 = State(
                            [self.s2_count,self.s2_count+1],
                            [1],
                            [self.s2_count],
                            scope=self.s2_scope_idx
                        )
                    self.SID_to_s2[self.s2_count] = deepcopy(new_s2)
                    node.children.append(self.SID_to_s2[self.s2_count])
                    self.s1_to_s2s[s1].append(self.SID_to_s2[self.s2_count])
                    self.s2_count += 1
            elif isinstance(node, Max) or isinstance(node, Sum):
                for child in node.children:
                    q.put(child)
        return spmn

    # TODO replace this by using a placeholder for s2 as last infoset in partial order,
    #  --- then just replace that placeholder using method above
    def assign_s2(self, spmn):
        s1 = spmn.children[0]
        q = queue.Queue()
        q.put(spmn)
        while not q.empty():
            node = q.get()
            if isinstance(node, Max) or isinstance(node, Sum):
                for child in node.children:
                    if isinstance(node, Max) or isinstance(node, Sum) or isinstance(node, Product):
                        q.put(child)
                    else:
                        node.children.remove(child)
                        new_s2 = State(
                                [self.s2_count,self.s2_count+1],
                                [1],
                                [self.s2_count],
                                scope=self.s2_scope_idx
                            )
                        self.SID_to_s2[self.s2_count] = new_s2
                        node.children.append(Product(
                                children=[
                                    child,
                                    new_s2
                                ]
                            ))
                        self.s2_count += 1
            elif isinstance(node, Product):
                is_terminal = True
                for child in node.children:
                    if isinstance(child, Max) or isinstance(child, Sum):
                        is_terminal = False
                if is_terminal:
                    new_s2 = State(
                            [self.s2_count,self.s2_count+1],
                            [1],
                            [self.s2_count],
                            scope=self.s2_scope_idx
                        )
                    self.SID_to_s2[s2_count] = new_s2
                    self.s1_to_s2s[s1].append(new_s2)
                    node.children.append(new_s2)
                    self.s2_count += 1
                else:
                    for child in node.children:
                        q.put(child)
        return spmn

    def update_s_nodes(self):
        nodes = get_nodes_by_type(self.spmn.spmn_structure)
        for node in nodes:
            if type(node)==State:
                bin_repr_points = list(range(self.s2_count))
                breaks = list(range(self.s2_count+1))
                densities = []
                for i in range(self.s2_count):
                    if i in node.bin_repr_points:
                        densities.append(node.densities[node.bin_repr_points.index(i)])
                    else:
                        densities.append(0)
                node.bin_repr_points = bin_repr_points
                node.breaks = breaks
                node.densities = densities

    def set_new_s1_vals(self, train_data, last_step_with_SID_idx, can_get_next_SID):
        nans=np.empty((train_data.shape[0],train_data.shape[1],1))
        nans[:] = np.nan
        # s1 at t is s2 at t-1
        train_data_s2 = np.concatenate((train_data,nans),axis=2)
        prev_step_data = train_data_s2[
                np.arange(train_data.shape[0]),
                last_step_with_SID_idx
            ]
        prev_SIDs = np.unique(prev_step_data[:,0]).astype(int)
        relevant_branches = list()
        for SID in prev_SIDs:
            if SID in self.SID_to_branch:
                branch = self.SID_to_branch[SID]
                if not branch in relevant_branches:
                    relevant_branches.append(branch)
        for branch in relevant_branches:
            branch_data_indices = np.arange(train_data.shape[0])[
                    np.logical_and(
                            can_get_next_SID,
                            np.isin(prev_step_data[:,0], self.branch_to_SIDs[branch])
                        )
                ]
            branch = assign_ids(branch)
            x = train_data_s2[
                    branch_data_indices,
                    last_step_with_SID_idx[branch_data_indices]
                ]
            #print(f"x.shape: {x.shape}")
            #print(f"branch: {branch}")
            #print(self.branch_to_SIDs[branch])
            new_SIDs = mpe(
                    branch,
                    train_data_s2[
                            branch_data_indices,
                            last_step_with_SID_idx[branch_data_indices]
                        ]
                )[:,self.s2_scope_idx]
            if np.any(np.isnan(new_SIDs)):
                print("\nfound nan SID assignment\n\n")
                new_SIDs[np.isnan(new_SIDs)] = -1
            train_data[
                    branch_data_indices,
                    last_step_with_SID_idx[branch_data_indices]+1,
                    0
                ] = new_SIDs
        self.spmn.spmn_structure = assign_ids(self.spmn.spmn_structure)
        #new_s1s = mpe(spmn_t_structure, prev_step_data)[:,len(scopeVars)]
        #train_data[:,t,0] = new_s1s
        return train_data



    def matches_state_branch(self, branch, train_data, SID_indices,
            last_step_with_SID_idx):
        branch_SIDs = self.branch_to_SIDs[branch]
        branch_SIDs_in_data = np.isin(train_data[:,:,0].astype(int),branch_SIDs)
        branch_sequence_indices = np.any(branch_SIDs_in_data, axis=1)
        branch_step_indices = np.argmax(branch_SIDs_in_data, axis=1)
        for i in range(0,self.horizon):
            # select only sequences with sufficient remaining depth
            branch_sequence_indices_i = np.logical_and(
                    branch_sequence_indices,
                    (branch_step_indices+i)<self.problem_depth
                )
            SID_indices_i = np.logical_and(
                    SID_indices,
                    (last_step_with_SID_idx+i)<self.problem_depth
                )
            branch_data = train_data[
                    branch_sequence_indices_i,
                    branch_step_indices[branch_sequence_indices_i]+i
                ]
            try:
                newSID_data = train_data[
                        SID_indices_i,
                        last_step_with_SID_idx[SID_indices_i]+i
                    ]
            except:
                print("SID_indices_i.dtype:\t"+str(SID_indices_i.dtype))
                print("last_step_with_SID_idx.dtype:\t"+str(last_step_with_SID_idx.dtype))
                print("i:\t"+str(i))
                print("SID_indices.shape"+str(SID_indices.shape))
                print("SID_indices_i:\t"+str(SID_indices_i))
            # use SIDs of initial step
            SIDs = np.append(
                        train_data[
                                SID_indices_i,
                                last_step_with_SID_idx[SID_indices_i]
                            ][:,0],
                        train_data[
                                branch_sequence_indices_i,
                                branch_step_indices[branch_sequence_indices_i]
                            ][:,0]
                    )
            # print("newSID_data: "+str(newSID_data.shape)+"\n"+str(newSID_data[:10]))
            # print("branch_data: "+str(branch_data.shape)+"\n"+str(branch_data[:10]))
            chi2_data = np.append(newSID_data,branch_data,axis=0)
            if chi2_data.shape[0] == 0: continue
            chi2_data = np.delete(chi2_data,[0]+self.dec_indices,axis=1) # remove decision and SID values
            # print("dec_indices:\t"+str(self.dec_indices))
            # print("chi2_data.shape:\t"+str(chi2_data.shape))
            # print("SID_indices count:\t"+str(np.sum(SID_indices)))
            # print("branch_sequence_indices count:\t"+str(np.sum(branch_sequence_indices)))
            chi2_data -= np.amin(chi2_data,axis=0) # ensure values are positive for chi2
            # print("chi2_data: "+str(chi2_data.shape)+"\n"+str(chi2_data[:10]))
            # look for correlations between SID and data
            min_chi2_pvalue = np.nanmin(chi2(
                    chi2_data,
                    SIDs
                )[1])
            if min_chi2_pvalue < self.chi2_threshold:
                #print(f"match failed; min_chi2_pvalue:\t{min_chi2_pvalue}")
                return False
        print("match found!")
        return True












    ################################ learn #####################################
    def learn_s_rspmn(self, data, plot = False):
        print("self.chi2_threshold:\t"+str(self.chi2_threshold))
        nans=np.empty((data.shape[0],data.shape[1],1))
        nans[:] = np.nan
        train_data = np.concatenate((nans,data),axis=2)
        train_data[:,0,0]=0
        # merge sequence steps based on horizon
        train_data_h = self.get_horizon_train_data(data, 2)
        # s1 for step 1 is 0
        train_data_h[:,0,0]=0

        partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = self.get_horizon_params(
                self.partialOrder, self.decNode, self.utilNode, self.scopeVars, self.meta_types, 2
            )

        start_time = time.perf_counter()
        spmn0 = SPMN(
                partialOrder_h,
                decNode_h,
                utilNode_h,
                scopeVars_h,
                meta_types_h,
                cluster_by_curr_information_set=True,
                util_to_bin = False
            )
        if True:
            print("start learning spmn0")
            spmn0_structure = spmn0.learn_spmn(train_data_h[:,0], self.chi2_threshold)
            spmn0_stoptime = time.perf_counter()
            spmn0_runtime = spmn0_stoptime - start_time
            print("learining spmn0 runtime:\t" + str(spmn0_runtime))
            print("spmn0 nodes:\t" + str(len(get_nodes_by_type(spmn0_structure))))
            file = open(f"spmn_0.pkle",'wb')
            import pickle
            pickle.dump(spmn0_structure, file)
            file.close()
        else:
            file = open(f"spmn_0.pkle",'rb')
            import pickle
            spmn0_structure = pickle.load(file)
            file.close()


        if plot:
            from spn.io.Graphics import plot_spn
            plot_spn(spmn0_structure, "test_spmn0.png")

        self.s2_count = 1
        spmn0_structure = self.replace_nextState_with_s2(spmn0_structure) # s2 is last scope index
        spmn0_structure = assign_ids(spmn0_structure)
        spmn0_structure = rebuild_scopes_bottom_up(spmn0_structure)
        # update state nodes to contain probabilities for all state values
        self.SID_to_branch[0] = spmn0_structure
        self.branch_to_SIDs[spmn0_structure] = [0]
        self.s1_node_to_SIDs[spmn0_structure.children[0]] = [0]

        if plot:
            from spn.io.Graphics import plot_spn
            plot_spn(spmn0_structure,"test_spmn0_with_s2.png")

        spmn_t = SPMN(
                self.partialOrder,
                self.decNode,
                self.utilNode,
                self.scopeVars,
                self.meta_types,
                cluster_by_curr_information_set=True,
                util_to_bin = False
            )
        spmn_t_structure = Sum(weights=[1],children=[spmn0_structure])
        spmn_t_structure = assign_ids(spmn_t_structure)
        spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
        spmn_t.spmn_structure = spmn_t_structure
        self.spmn = spmn_t
        self.update_s_nodes()

        done = False
        total_pushing_SIDs_time = 0
        total_time_learning_structures = 0
        total_time_matching = 0
        while True:
            current_total_runtime = time.perf_counter() - start_time
            print("\n\nruntime so far:\t" + str(current_total_runtime)+"\tnum_branches:\t"+str(len(self.spmn.spmn_structure.children)))
            percent_time_pushing_SIDs = (total_pushing_SIDs_time / current_total_runtime)*100
            print("percent_time_pushing_SIDs:\t%.2f" % percent_time_pushing_SIDs)
            percent_time_learning_structures = (total_time_learning_structures / current_total_runtime)*100
            print("percent_time_learning_structures:\t%.2f" % percent_time_learning_structures)
            percent_time_matching = (total_time_matching / current_total_runtime)*100
            print("percent_time_matching:\t%.2f" % percent_time_matching)
            # push sequences forward through the existing structure until they all
            #   reach an SID which has not yet been linked to a branch.
            start_pushing_SIDs_time = time.perf_counter()
            while True:
                last_step_with_SID_idx = (np.argmax(np.isnan(train_data[:,:,0]), axis=1)-1).astype(int)
                last_step_with_SID_idx[last_step_with_SID_idx==-1] = self.problem_depth-1
                remaining_steps = np.sum(np.isnan(train_data[:,:,0]),axis=1)
                last_step_already_modeled = np.isin(
                        train_data[
                                np.arange(train_data.shape[0]),
                                last_step_with_SID_idx
                            ][:,0],
                        list(self.SID_to_branch.keys())
                    )
                can_get_next_SID = np.logical_and(last_step_already_modeled, remaining_steps > 0)
                # if any sequences' last processed step have SIDs which match to
                #   existing branches and have steps remaining:
                if np.any(can_get_next_SID) and np.any(np.isnan(train_data[:,:,0])):
                    # get the next SID
                    train_data = self.set_new_s1_vals(train_data, last_step_with_SID_idx, can_get_next_SID)
                    train_data_h[:,:,0] = train_data[:,:,0]
                else:
                    break
            pushing_SIDs_time = time.perf_counter() - start_pushing_SIDs_time
            total_pushing_SIDs_time += pushing_SIDs_time
            # once all sequences have reached a stopping point, find the unlinked
            #   SID with the most data waiting behind it.
            max_data_val = 0
            max_val_SID = None
            max_val_SID_indices = None
            unmatched_SID = False
            for SID in range(1, self.s2_count):
                if not SID in self.SID_to_branch:
                    unmatched_SID = True
                    SID_indices = train_data[
                                np.arange(train_data.shape[0]),
                                last_step_with_SID_idx
                            ][:,0]==SID
                        #np.arange(train_data.shape[0])[
                        #     train_data[
                        #             np.arange(train_data.shape[0]),
                        #             last_step_with_SID_idx
                        #         ][:self.s2_scope_idx]==SID
                        # ]
                    SID_data_val = np.sum(remaining_steps[SID_indices]+1)
                    # print("SID_data_val:\t"+str(SID_data_val))
                    # if SID_data_val == 0:
                    #     print("0 val SID:\t"+str(SID))
                    #     print("self.SID_to_branch:\n"+str(self.SID_to_branch))
                    if SID_data_val > max_data_val:
                        max_data_val = SID_data_val
                        max_val_SID = SID
                        max_val_SID_indices = SID_indices
            if not np.any(last_step_with_SID_idx < self.problem_depth) or not unmatched_SID or max_data_val==0:
                break
            # look for an existing branch that adequately models the data corresponding
            #   to this SID
            matched = False
            print(f"\nstart matching for SID {max_val_SID}")
            print("max_data_val:\t"+str(max_data_val))
            print("max_val_SID_indices:\t"+str(max_val_SID_indices))
            start_matching_time = time.perf_counter()
            for branch in self.spmn.spmn_structure.children:
                if self.matches_state_branch(branch, train_data, max_val_SID_indices,
                        last_step_with_SID_idx):
                    self.branch_to_SIDs[branch].append(max_val_SID)
                    branch_SIDs = self.branch_to_SIDs[branch]
                    branch_s1_node = branch.children[0]
                    densities = branch_s1_node.densities
                    for SID in branch_SIDs:
                        densities[SID] = 1
                    branch_s1_node.densities = densities
                    # link s2 for new_val to this s1 node
                    self.SID_to_s2[max_val_SID].interface_links[branch_s1_node] = np.sum(max_val_SID_indices)
                    self.s1_node_to_SIDs[branch_s1_node] = branch_SIDs
                    self.SID_to_branch[max_val_SID] = branch
                    weights = []
                    for child in self.spmn.spmn_structure.children:
                        child_s1_vals = self.branch_to_SIDs[child]
                        count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                        for s1_val in child_s1_vals:
                            if s1_val == 0:
                                count_child += train_data.shape[0] # 1 starting state for each sequence
                            else:
                                count_child += self.SID_to_s2[s1_val].interface_links[child.children[0]]
                        prob_child = count_child / (self.samples * self.problem_depth)
                        weights.append(prob_child)
                    normalized_weights = np.array(weights) / np.sum(weights)
                    self.spmn.spmn_structure.weights = normalized_weights.tolist()
                    matched = True
                    # as each branch is created to model a different distribution,
                    #   we can expect that no further matches will be found.
                    break
            matching_time = time.perf_counter() - start_matching_time
            total_time_matching += matching_time
            start_time_learning_structure = time.perf_counter()
            if not matched:
                ################ < creating new branch for state   #############
                h = 2#self.horizon
                tdh = self.get_horizon_train_data(data, h)
                tdh[:,:,0] = train_data[:,:,0]
                while True:
                    new_spmn_data = tdh[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                    new_spmn_sl_data = new_spmn_data[~np.any(np.isnan(new_spmn_data),axis=1)]
                    if new_spmn_sl_data.shape[0] > 100:
                        break
                    elif h <= 2:
                        print(f"\n\th=1 for {max_val_SID}\n")
                        h = 1
                        new_spmn_sl_data = train_data[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                        print("new_spmn_sl_data.shape::\t"+str(new_spmn_sl_data.shape))
                        new_spmn_sl_data = np.concatenate((new_spmn_sl_data,np.ones((new_spmn_sl_data.shape[0],1))), axis=1)
                        print("new_spmn_sl_data.shape::\t"+str(new_spmn_sl_data.shape))
                        break
                    else:
                        h -= 1
                        tdh = self.get_horizon_train_data(data, h)
                        tdh[:,:,0] = train_data[:,:,0]
                new_spmn_em_data = train_data[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                em_nans = np.empty((new_spmn_em_data.shape[0],1))
                new_spmn_em_data = np.concatenate((new_spmn_em_data,em_nans),axis=1)
                partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = self.get_horizon_params(
                        self.partialOrder, self.decNode, self.utilNode, self.scopeVars, self.meta_types, h
                    )
                if h == 1:
                    partialOrder_h.append(["dummy"])
                    scopeVars_h.append("dummy")
                    meta_types_h.append(MetaType.DISCRETE)
                spmn_new_s1 = SPMN(
                        partialOrder_h,
                        decNode_h,
                        utilNode_h,
                        scopeVars_h,
                        meta_types_h,
                        cluster_by_curr_information_set=True,
                        util_to_bin = False
                    )
                branch_num = len(self.spmn.spmn_structure.children)
                percentage_of_data_sl = (new_spmn_sl_data.shape[0]/self.samples)*100
                percentage_of_data_em = (new_spmn_em_data.shape[0]/self.samples)*100
                print(f"\ncreating branch {branch_num} for SID {max_val_SID}, \npercentage of data for SL: {percentage_of_data_sl}%, \npercentage of data for EM: {percentage_of_data_em}%")
                remaining_data = np.sum(remaining_steps)
                print(f"total remaining data: {remaining_data}")
                # print("\nnew_spmn_data[:10]:\n"+str(new_spmn_data[:10]))
                # print("\nlast_step_with_SID_idx[:5]:\n"+str(last_step_with_SID_idx[:5]))
                # print("\ntrain_data[:5]:\n"+str(train_data[:5]))
                spmn_new_s1_structure = spmn_new_s1.learn_spmn(new_spmn_sl_data, self.chi2_threshold)
                if h > 1:
                    spmn_new_s1_structure = self.replace_nextState_with_s2(spmn_new_s1_structure)
                else:
                    print(f"h = 1 for SID {max_val_SID}")
                    spmn_new_s1_structure = self.replace_nextState_with_s2(spmn_new_s1_structure)
                    from spn.io.Graphics import plot_spn
                    plot_spn(spmn_new_s1_structure, "replaced_dummies.png")
                    # spmn_new_s1_structure = self.assign_s2(spmn_new_s1_structure)
                # print("perfoming EM optimization")
                # EM_optimization(spmn_new_s1_structure, new_spmn_em_data, iterations=1, skip_validation=True)
                branch_s1_node = spmn_new_s1_structure.children[0]
                self.SID_to_branch[max_val_SID] = spmn_new_s1_structure
                self.SID_to_s2[max_val_SID].interface_links[branch_s1_node] = np.sum(max_val_SID_indices)
                self.s1_node_to_SIDs[branch_s1_node] = [max_val_SID]
                self.branch_to_SIDs[spmn_new_s1_structure] = [max_val_SID]
                self.spmn.spmn_structure.children += [spmn_new_s1_structure]
                weights = []
                for child in self.spmn.spmn_structure.children:
                    child_s1_vals = self.branch_to_SIDs[child]
                    count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                    for s1_val in child_s1_vals:
                        if s1_val == 0:
                            count_child += train_data.shape[0] # 1 starting state for each sequence
                        else:
                            count_child += self.SID_to_s2[s1_val].interface_links[child.children[0]]
                    prob_child = count_child / (self.samples * self.problem_depth)
                    weights.append(prob_child)
                normalized_weights = np.array(weights) / np.sum(weights)
                self.spmn.spmn_structure.weights = normalized_weights.tolist()
                self.update_s_nodes()
                self.spmn.spmn_structure = assign_ids(self.spmn.spmn_structure)
                self.spmn.spmn_structure = rebuild_scopes_bottom_up(self.spmn.spmn_structure)
            time_learning_structure = time.perf_counter() - start_time_learning_structure
            total_time_learning_structures += time_learning_structure
        learn_s_rspmn_stoptime = time.perf_counter()
        learn_s_rspmn_runtime = learn_s_rspmn_stoptime - start_time
        print(f"learn_s_rspmn runtime: {learn_s_rspmn_runtime}")
        nodes = get_nodes_by_type(self.spmn.spmn_structure)
        if plot:
            from spn.io.Graphics import plot_spn
            plot_spn(self.spmn.spmn_structure, "test_s-rspmn.png")




























#################################### main ######################################

if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="crossing_traffic")
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--plot", default=False, type=bool)
    parser.add_argument("--apply_em", default=False)
    parser.add_argument("--use_chi2", default=True)
    parser.add_argument("--chi2_threshold", default=0.05, type=float)
    parser.add_argument("--likelihood_similarity_threshold", default=0.000000001, type=float)
    parser.add_argument("--likelihood_match", default=True)
    parser.add_argument("--deep_match", default=True)
    parser.add_argument("--horizon", default=2, type=int)
    parser.add_argument("--problem_depth", default=10, type=int)
    parser.add_argument("--samples", default=100000, type=int)
    parser.add_argument("--num_vars", default=17, type=int)#total number of columns in dataset

    args = parser.parse_args()

    rspmn = S_RSPMN(
                dataset = args.dataset,
                debug = args.debug==2,
                debug1 = args.debug>0,
                apply_em = args.apply_em,
                use_chi2 = args.use_chi2,
                chi2_threshold = args.chi2_threshold,
                likelihood_similarity_threshold = args.likelihood_similarity_threshold,
                likelihood_match = args.likelihood_match,
                deep_match = args.deep_match,
                horizon = args.horizon,
                problem_depth = args.problem_depth,
                samples = args.samples,
                num_vars = 14 if args.dataset == "crossing_traffic" else args.num_vars
            )

    df = pd.read_csv(
        f"data/{args.dataset}/{args.dataset}_{args.samples}x{args.problem_depth}.tsv",
        index_col=0, sep='\t',
        header=0 if args.dataset=="repeated_marbles" or args.dataset=="tiger" else None)
    data = df.values.reshape(args.samples,args.problem_depth,args.num_vars)

    if args.dataset == "crossing_traffic":
        decisions = data[:,:,12:16]
        decisions = np.concatenate((np.zeros((decisions.shape[0],decisions.shape[1],1)),decisions),axis=2)
        decisions = np.argmax(decisions,axis=2)
        data = np.concatenate(
                (
                    data[:,:,:12],
                    decisions.reshape(data.shape[0],-1,1),
                    data[:,:,-1].reshape(data.shape[0],-1,1)
                ),
                axis=2
            )

    rspmn.learn_s_rspmn(data, plot = args.plot)

    date = str(datetime.date(datetime.now()))[-5:].replace('-','')
    hour = str(datetime.time((datetime.now())))[:2]
    file = open(f"data/{args.dataset}/rspmn_{date}_{hour}.pkle",'wb')
    import pickle
    pickle.dump(rspmn, file)
    file.close()

    #from spn.algorithms.MEU import rmeu
    input_data = np.array([0]+[np.nan]*(args.num_vars+1))
    for i in range(1,args.problem_depth+1):
        print(f"rmeu for depth {i}:\t"+str(rmeu(rspmn.spmn, input_data, depth=i)))



def get_action(branch, SID, dec_indices, num_vars=17):
    for i in range(len(dec_indices)):
        input_data = np.array([[np.nan]*(num_vars-len(dec_indices))+[0]*len(dec_indices)+[np.nan,SID]])
        input_data[0][(dec_indices[i])] = 1
        if likelihood(branch, input_data) > 0.000001:
            return i
    return "noop"












def get_branch_and_decisions_to_s2(rspmn_root):
    branch_and_decisions_to_s2 = dict()
    for branch in rspmn_root.children:
        queue = branch.children[1:]
        fill_branch_and_decisions_to_s2(branch_and_decisions_to_s2, queue, [branch])
    branch_to_decisions_to_s2s = dict()
    for branch_and_decisions, s2 in branch_and_decisions_to_s2.items():
        branch = branch_and_decisions[0]
        decision_path = branch_and_decisions[1:]
        if branch in branch_to_decisions_to_s2s:
            branch_to_decisions_to_s2s[branch][decision_path] = s2
        else:
            branch_to_decisions_to_s2s[branch] = {decision_path: s2}
    return branch_to_decisions_to_s2s

def fill_branch_and_decisions_to_s2(branch_and_decisions_to_s2, queue, path):
    while len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node, Max):
            for i in range(len(node.dec_values)):
                dec_val_i = node.dec_values[i]
                child_i = node.children[i]
                fill_branch_and_decisions_to_s2(
                        branch_and_decisions_to_s2,
                        [child_i],
                        path+[dec_val_i]
                    )
        elif isinstance(node, State):
            if tuple(path) in branch_and_decisions_to_s2:
                branch_and_decisions_to_s2[tuple(path)] += [node]
            else:
                branch_and_decisions_to_s2[tuple(path)] = [node]
        elif isinstance(node, Product) or isinstance(node, Sum):
            for child in node.children:
                queue.append(child)

def clear_caches(rspmn):
    del rspmn.branch_to_decisions_to_s2s
    del rspmn.branch_and_decisions_to_meu
    del rspmn.branch_and_decisions_and_s2_to_likelihood
    del rspmn.branch_and_depth_to_rmeu




def rmeu(rspmn, input_data, depth):
    assert not np.isnan(input_data[0]), "starting SID (input_data[0]) must be defined."
    root = rspmn.spmn.spmn_structure
    branch = rspmn.SID_to_branch[input_data[0]]
    branch = assign_ids(branch)
    from spn.algorithms.MEU import meu
    # set up caches
    if not hasattr(rspmn,"branch_to_decisions_to_s2s"):
        branch_to_decisions_to_s2s = get_branch_and_decisions_to_s2(root)
        setattr(rspmn,"branch_to_decisions_to_s2s",branch_to_decisions_to_s2s)
    if not hasattr(rspmn,"branch_and_decisions_to_meu"):
        branch_and_decisions_to_meu = dict()
        setattr(rspmn,"branch_and_decisions_to_meu",branch_and_decisions_to_meu)
    if not hasattr(rspmn,"branch_and_decisions_and_s2_to_likelihood"):
        branch_and_decisions_and_s2_to_likelihood = dict()
        setattr(rspmn,"branch_and_decisions_and_s2_to_likelihood",
            branch_and_decisions_and_s2_to_likelihood)
    if not hasattr(rspmn,"branch_and_depth_to_rmeu"):
        branch_and_depth_to_rmeu = dict()
        setattr(rspmn,"branch_and_depth_to_rmeu",branch_and_depth_to_rmeu)
    max_EU = None
    # if unconditioned meu for this state branch and depth has already been cached, just return the cached value
    if np.all(np.isnan(input_data[1:])):
        if (branch, depth) in rspmn.branch_and_depth_to_rmeu:
            return rspmn.branch_and_depth_to_rmeu[(branch, depth)]
        elif depth == 1:
            max_EU = meu(branch, np.array([input_data]))
            rspmn.branch_and_depth_to_rmeu[(branch, depth)] = max_EU
            return max_EU
    elif depth == 1:
        return meu(branch, np.array([input_data]))
    for decision_path, s2s in rspmn.branch_to_decisions_to_s2s[branch].items():
        path_data = deepcopy(input_data)
        ############### for unconditioned inputs, we can use caches ############
        if np.all(np.isnan(path_data[1:])):
            for i in range(len(decision_path)):
                path_data[rspmn.dec_indices[i]] = decision_path[i]
            if (branch,decision_path) in rspmn.branch_and_decisions_to_meu:
                path_EU = rspmn.branch_and_decisions_to_meu[(branch,decision_path)]
            else:
                branch = assign_ids(branch)
                path_EU = meu(branch, np.array([path_data])).reshape(-1)
                rspmn.branch_and_decisions_to_meu[(branch,decision_path)] = path_EU
            future_EU = 0
            s2_norm = 0
            for s2 in s2s:
                SID = np.argmax(s2.densities).astype(int)
                path_data[rspmn.s2_scope_idx] = SID
                if (branch,decision_path,s2) in rspmn.branch_and_decisions_and_s2_to_likelihood:
                    s2_likelihood = rspmn.branch_and_decisions_and_s2_to_likelihood[(branch,decision_path,s2)]
                else:
                    s2_likelihood = likelihood(branch,np.array([path_data])).reshape(-1)
                    rspmn.branch_and_decisions_and_s2_to_likelihood[(branch,decision_path,s2)] = s2_likelihood
                if SID in rspmn.SID_to_branch:
                    s2_norm += s2_likelihood
                    next_data = np.array([SID]+[np.nan]*(rspmn.num_vars+1))
                    future_val = rmeu(rspmn, next_data, depth-1)
                    if future_val:
                        future_EU += future_val * s2_likelihood
                        s2_norm += s2_likelihood
            if s2_norm != 0:
                future_EU /= s2_norm
                #print(f"depth:\t{depth}\tfuture_EU:\t{future_EU}")
                total_path_EU = path_EU + future_EU
                if not max_EU or total_path_EU > max_EU: max_EU = total_path_EU
        ############## for conditioned inputs, we canNOT use caches ############
        else:
            for i in range(len(decision_path)):
                if not (input_data[rspmn.dec_indices[i]] == decision_path[i]):
                    continue # only consider paths that match the input
                path_data[rspmn.dec_indices[i]] = decision_path[i]
            branch = assign_ids(branch)
            path_EU = meu(branch, np.array([path_data])).reshape(-1)
            future_EU = 0
            s2_norm = 0
            for s2 in s2s:
                SID = np.argmax(s2.densities).astype(int)
                path_data[rspmn.s2_scope_idx] = SID
                s2_likelihood = likelihood(branch,np.array([path_data])).reshape(-1)
                next_data = np.array([SID]+[np.nan]*(rspmn.num_vars+1))
                if SID in rspmn.SID_to_branch:
                    future_val = rmeu(rspmn, next_data, depth-1)
                    if future_val:
                        future_EU += future_val * s2_likelihood
                        s2_norm += s2_likelihood
            if s2_norm != 0:
                future_EU = future_EU / s2_norm
                total_path_EU = path_EU + future_EU
                if not max_EU or total_path_EU > max_EU: max_EU = total_path_EU
    root = assign_ids(root)
    return max_EU

#clear_caches(rspmn)
#rmeu(rspmn, input_data, 2)
