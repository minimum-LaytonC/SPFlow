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

dataset = "crossing_traffic"
plot = True
apply_em = False
use_chi2 = True
chi2_threshold = 0.4
likelihood_similarity_threshold = 0.95
horizon = 2

if dataset == "repeated_marbles":
    df = pd.DataFrame.from_csv("data/"+dataset+"/repeated_marbles_10000x20.tsv", sep='\t')
    data = df.values.reshape(10000,20,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['draw'],['result','util']]
    decNode=['draw']
    utilNode=['util']
    scopeVars=['s1','draw','result','util']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "tiger":
    df = pd.DataFrame.from_csv("data/"+dataset+"/reverse_tiger_100000x5.tsv", sep='\t')
    data = df.values.reshape(100000,5,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['observation'],['action'],['reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','observation','action','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "frozen_lake":
    df = pd.DataFrame.from_csv("data/"+dataset+"/frozen_lake_1000000x10.tsv", sep='\t')
    data = df.values.reshape(1000000,10,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['action'],['observation','reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','action','observation','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "nchain":
    df = pd.DataFrame.from_csv("data/"+dataset+"/nchain_100000x10.tsv", sep='\t')
    data = df.values.reshape(100000,10,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['action'],['observation','reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','action','observation','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "elevators":
    df = pd.DataFrame.from_csv("data/"+dataset+"/elevators_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,10)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'close-door',
            'move-current-dir',
            'open-door-going-up',
            'open-door-going-down',
        ]
    obs = [
            'person-in-elevator-going-down',
            'person-waiting-1',
            'person-waiting-2',
            'person-waiting-3',
            'person-in-elevator-going-up',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs+['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,8,9,1,2,3,4,10]
    #train_data[:] = train_data[:,permutation]
elif dataset == "skill_teaching":
    df = pd.DataFrame.from_csv("data/"+dataset+"/skill_teaching_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,9)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'giveHint-1',
            'giveHint-2',
            'askProb-1',
            'askProb-2',
        ]
    obs = [
            'answeredRightObs-1',
            'answeredRightObs-2',
            'updateTurnObs-1',
            'updateTurnObs-2',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs,['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,8,1,2,3,4,9]
    #train_data[:] = train_data[:,permutation]
elif dataset == "crossing_traffic":
    df = pd.DataFrame.from_csv("data/"+dataset+"/crossing_traffic_1000000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(1000000,10,8)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'move-east',
            'move-north',
            'move-south',
            'move-west'
        ]
    obs = [
            'arrival-max-xpos-1',
            'arrival-max-xpos-2',
            'arrival-max-xpos-3'
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs,['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,1,2,3,4,8]
    #train_data[:] = train_data[:,permutation]
elif dataset == "game_of_life":
    df = pd.DataFrame.from_csv("data/"+dataset+"/game_of_life_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,19)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'set-1-1',
            'set-1-2',
            'set-1-3',
            'set-2-1',
            'set-2-2',
            'set-2-3',
            'set-3-1',
            'set-3-2',
            'set-3-3',
        ]
    obs = [
            'surround-obs-1-1',
            'surround-obs-1-2',
            'surround-obs-1-3',
            'surround-obs-2-1',
            'surround-obs-2-2',
            'surround-obs-2-3',
            'surround-obs-3-1',
            'surround-obs-3-2',
            'surround-obs-3-3',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs,['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,19]
    #train_data[:] = train_data[:,permutation]
elif dataset == "tamarisk":
    df = pd.DataFrame.from_csv("data/"+dataset+"/tamarisk_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,17)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'restore-1',
            'restore-2',
            'restore-3',
            'restore-4',
            'eradicate-1',
            'eradicate-2',
            'eradicate-3',
            'eradicate-4',
        ]
    obs = [
            'tamarisk-at-reach-obs-1',
            'tamarisk-at-reach-obs-2',
            'tamarisk-at-reach-obs-3',
            'tamarisk-at-reach-obs-4',
            'native-at-reach-obs-1',
            'native-at-reach-obs-2',
            'native-at-reach-obs-3',
            'native-at-reach-obs-4',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs,['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,19]
    #train_data[:] = train_data[:,permutation]
elif dataset == "traffic_control":
    df = pd.DataFrame.from_csv("data/"+dataset+"/traffic_control_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,13)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'advance[$ia3a3]',
            'advance[$ia3a6]',
            'advance[$ia6a3]',
            'advance[$ia6a6]',
        ]
    obs = [
            'traffic-at-stopline-EW-obs[$ia3a3]',
            'traffic-at-stopline-EW-obs[$ia3a6]',
            'traffic-at-stopline-EW-obs[$ia6a3]',
            'traffic-at-stopline-EW-obs[$ia6a6]',
            'traffic-at-stopline-NS-obs[$ia3a3]',
            'traffic-at-stopline-NS-obs[$ia3a6]',
            'traffic-at-stopline-NS-obs[$ia6a3]',
            'traffic-at-stopline-NS-obs[$ia6a6]',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+decNode+obs+['reward']
    partialOrder = [['s1']]+[[x] for x in decNode]+[obs,['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,19]
    #train_data[:] = train_data[:,permutation]


def get_horizon_train_data(data, horizon):
    # following line should concat each timestep with the next 'horizon' timesteps
    train_data_h = np.concatenate([data[:,i:data.shape[1]-horizon+i+1] for i in range(horizon)],axis=2)
    # add nans for s1
    nans_h=np.empty((train_data_h.shape[0],train_data_h.shape[1],1))
    nans_h[:] = np.nan
    train_data_h = np.concatenate((nans_h,train_data_h),axis=2)
    return train_data_h

# merge sequence steps based on horizon
train_data_h = get_horizon_train_data(data, horizon)
# s1 for step 1 is 0
train_data_h[:,0,0]=0

def get_horizon_params(partialOrder, decNode, utilNode, scopeVars, meta_types, horizon):
    partialOrder_h = [] + partialOrder
    for i in range(1,horizon):
        partialOrder_h += [[var+"_t+"+str(i) for var in s] for s in partialOrder[1:]]
    decNode_h = decNode+[decNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(decNode))]
    utilNode_h = utilNode+[utilNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(utilNode))]
    scopeVars_h = scopeVars + [var+"_t+"+str(i) for var in scopeVars[1:] for i in range (1,horizon)]
    meta_types_h = meta_types+meta_types[1:]*(horizon-1)
    return partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h

partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = get_horizon_params(
        partialOrder, decNode, utilNode, scopeVars, meta_types, horizon
    )

spmn0 = SPMN(
        partialOrder_h,
        decNode_h,
        utilNode_h,
        scopeVars_h,
        meta_types_h,
        cluster_by_curr_information_set=True,
        util_to_bin = False
    )
spmn0_structure = spmn0.learn_spmn(train_data_h[:,0], chi2_threshold)

if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn0_structure, "plots/"+dataset+"/spmn0.png")


s2_dict = dict()
s1_to_s2s = dict()

import queue
def replace_nextState_with_s2(spmn,s2_scope_idx,s2_count=1, s2_dict=s2_dict, s1_to_s2s=s1_to_s2s):
    s1 = spmn.children[0]
    s1_to_s2s[s1] = list()
    scope_t1 = {i for i in range(s2_scope_idx)}
    q = q = queue.Queue()
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
                        [s2_count,s2_count+1],
                        [1],
                        [s2_count],
                        scope=s2_scope_idx
                    )
                node.children.append(new_s2)
                s2_dict[s2_count] = new_s2
                s1_to_s2s[s1].append(new_s2)
                s2_count += 1
        elif isinstance(node, Max) or isinstance(node, Sum):
            for child in node.children:
                q.put(child)
    return spmn, s2_count

# TODO replace this by using a placeholder for s2 as last infoset in partial order,
#  --- then just replace that placeholder with method above
import queue
def assign_s2(spmn,s2_scope_idx,s2_count=1, s2_dict=s2_dict):
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
                            [s2_count,s2_count+1],
                            [1],
                            [s2_count],
                            scope=s2_scope_idx
                        )
                    s2_dict[s2_count] = new_s2
                    node.children.append(Product(
                            children=[
                                child,
                                new_s2
                            ]
                        ))
                    s2_count += 1
        elif isinstance(node, Product):
            is_terminal = True
            for child in node.children:
                if isinstance(child, Max) or isinstance(child, Sum):
                    is_terminal = False
            if is_terminal:
                new_s2 = State(
                        [s2_count,s2_count+1],
                        [1],
                        [s2_count],
                        scope=s2_scope_idx
                    )
                s2_dict[s2_count] = new_s2
                s1_to_s2s[s1].append(new_s2)
                node.children.append(new_s2)
                s2_count += 1
            else:
                for child in node.children:
                    q.put(child)
    return spmn, s2_count

def update_s_nodes(spmn,s2_scope_idx,s2_count):
    # TODO caching state nodes for this would speed things up
    nodes = get_nodes_by_type(spmn)
    for node in nodes:
        if type(node)==State:
            bin_repr_points = list(range(s2_count))
            breaks = list(range(s2_count+1))
            densities = []
            for i in range(s2_count):
                if i in node.bin_repr_points:
                    densities.append(node.densities[node.bin_repr_points.index(i)])
                else:
                    densities.append(0)
            node.bin_repr_points = bin_repr_points
            node.breaks = breaks
            node.densities = densities
    return spmn

# add unique state identifier nodes for terminal branches of the spmn
spmn0_structure, s2_count = replace_nextState_with_s2(spmn0_structure, len(scopeVars), s2_count=1) # s2 is last scope index
spmn0_structure = assign_ids(spmn0_structure)
spmn0_structure = rebuild_scopes_bottom_up(spmn0_structure)
# update state nodes to contain probabilities for all state values
spmn0_structure = update_s_nodes(spmn0_structure,len(scopeVars),s2_count)


if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn0_structure,"plots/"+dataset+"/spmn0_with_s2.png")















#################### < creating template network    ####################
# create template network by adding a sum node with state branches as children
spmn_t = SPMN(
        partialOrder,
        decNode,
        utilNode,
        scopeVars,
        meta_types,
        cluster_by_curr_information_set=True,
        util_to_bin = False
    )

spmn_t_structure = Sum(weights=[1],children=[spmn0_structure])
spmn_t_structure = assign_ids(spmn_t_structure)
spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
####################    creating template network /> ####################

def set_new_s1_vals(train_data, spmn_t_structure, val_to_s_branch, s2_dict):
    nans=np.empty((train_data.shape[0],train_data.shape[1],1))
    nans[:] = np.nan
    # s1 at t is s2 at t-1
    train_data_s2 = np.concatenate((train_data,nans),axis=2)
    prev_step_data = train_data_s2[:,t-1]
    prev_s1s = np.unique(prev_step_data[:,0]).astype(int)
    for s1_val in prev_s1s:
        if s1_val == 0:
            state_structure = spmn_t_structure.children[0]
            state_structure = assign_ids(state_structure)
        else:
            linked_branches = val_to_s_branch[s1_val]
            counts = np.array(list(s2_dict[s1_val].interface_links.values()))
            child_weights = (counts/np.sum(counts)).tolist()
            state_structure = Sum(weights=child_weights,children=linked_branches)
            state_structure = assign_ids(state_structure)
            state_structure = rebuild_scopes_bottom_up(state_structure)
        new_s1s = mpe(state_structure, prev_step_data[prev_step_data[:,0]==s1_val])[:,len(scopeVars)]
        train_data[train_data[:,t-1,0]==s1_val,t,0] = new_s1s
    spmn_t_structure = assign_ids(spmn_t_structure)
    spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
    #new_s1s = mpe(spmn_t_structure, prev_step_data)[:,len(scopeVars)]
    #train_data[:,t,0] = new_s1s
    return train_data

def get_branch_s1_vals(branch):
    s1_node_vals = np.array(branch.children[0].bin_repr_points)
    s1_node_nonzero = np.ceil(branch.children[0].densities).astype(bool)
    #branch_s1_vals is the set of states this branch represents
    return list(s1_node_vals[s1_node_nonzero].astype(int))


def matches_state_branch(new_val, branch, spmn_t_structure, train_data, train_data_h, mean_branch_likelihoods, use_chi2, chi2_threshold,
        likelihood_similarity_threshold, s1_to_s2s, val_to_s_branch, t, d):
    train_data_unrolled = train_data[:,:t+1].reshape((-1,train_data.shape[2]))
    likelihood_train_data = train_data_unrolled_t[train_data_unrolled_t[:,0]==new_val]
    if likelihood_train_data.shape[0] < 1: # TODO This shouldn't ever happen?
        print("\n\nfound 0 length new_val data slice for new_val == " + str(new_val))
        continue
    # adding s2s as nan to satisfy code
    nans_lh_data=np.empty((likelihood_train_data.shape[0],1))
    nans_lh_data[:] = np.nan
    likelihood_train_data = np.concatenate((likelihood_train_data,nans_lh_data),axis=1)
    # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
    likelihood_train_data[:,0] = np.nan
    branch_s1_vals = get_branch_s1_vals(branch)
    print("\n\tstate, branch_index:\t"+str(new_val)+",\t"+str(spmn_t_structure.children.index(branch)))
    print("\tbranch_s1_vals:\t"+str(branch_s1_vals))
    testing_s1_vals = branch_s1_vals + [new_val]
    mask = np.isin(train_data_unrolled[:,0],testing_s1_vals)
    # select data corresponding to this node's states and the new state
    train_data_s1_selected = train_data_unrolled[mask]
    if use_chi2:
        # look for correlations between state and the other values
        min_chi2_pvalue = np.min(chi2(
                np.abs(np.delete(train_data_s1_selected,0,axis=1)),
                train_data_s1_selected[:,0]
            )[1])
    if branch in mean_branch_likelihoods:
        mean_likelihood_branch = mean_branch_likelihoods[branch]
    else:
        likelihood_data_branch = train_data_unrolled[
                np.isin(train_data_unrolled[:,0],branch_s1_vals)
            ]
        nans_lh_data=np.empty((likelihood_data_branch.shape[0],1))
        nans_lh_data[:] = np.nan
        likelihood_data_branch = np.concatenate((likelihood_data_branch,nans_lh_data),axis=1)
        # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
        likelihood_data_branch[:,0] = np.nan
        print("\t< start calculating likelihood for branch "+str(spmn_t_structure.children.index(branch)))
        mean_likelihood_branch = np.mean(likelihood(branch, likelihood_data_branch))
        print("\tend calculating likelihood for branch "+str(spmn_t_structure.children.index(branch))+ " >")
        mean_branch_likelihoods[branch] = mean_likelihood_branch
    likelihood_new = likelihood(branch, likelihood_train_data)
    mean_likelihood_new = np.mean(likelihood_new)
    min_likelihood_new = np.min(likelihood_new) if likelihood_train_data.shape[0] > 0 else np.nan
    print("\tmean_likelihood similarity:\t" + str(mean_likelihood_new/mean_likelihood_branch))
    print("\tmin_likelihood_new:\t"+str(min_likelihood_new))
    if use_chi2:
        print("\tmin_chi2_pvalue:\t" + str(min_chi2_pvalue))
    if (use_chi2 and min_chi2_pvalue < chi2_threshold)\
            or mean_likelihood_new/mean_likelihood_branch < likelihood_similarity_threshold \
            or min_likelihood_new < (1/10**10):
        # if s1 is correlated with any other variables,
        # then the new value is a functionally different state
        return False
    elif d > 1:
        branch_s2s = s1_to_s2s[branch.children[0]]
        data_copy = deepcopy(train_data)
        set_new_s1_vals(data_copy, branch, val_to_s_branch, s2_dict)
        for s2 in branch_s2s:
            if len(s2.interface_links) > 0:
                next_branch = list(s2.interface_links.keys())[0]

                if not matches_state_branch(new_val, next_branch, spmn_t_structure,
                        train_data, train_data_h, mean_branch_likelihoods, # TODO need to update s1s for t+1 and pass data by value
                        use_chi2, chi2_threshold, likelihood_similarity_threshold,
                        s1_to_s2s, val_to_s_branch, t+1, d-1):
                    return False
            else:
                # TODO do chi2 testing over data[:,t+1:t+d]
                train_data_h_unrolled = train_data_h[:,t].reshape((-1,train_data_h.shape[2]))
                mask = np.isin(train_data_h_unrolled[:,0],testing_s1_vals)
                train_data_s1_selected = train_data_unrolled[mask]
                min_chi2_pvalue = np.min(chi2(
                        np.abs(np.delete(train_data_s1_selected,0,axis=1)),
                        train_data_s1_selected[:,0]
                    )[1])
    return True




# learn new sub spmn branches based on state values
s1_vals = {0}
val_to_s_branch = dict()
val_to_s_branch[0]=[spmn_t_structure.children[0]]
mean_branch_likelihoods = dict()
for t in range(1, train_data.shape[1]):
    print("\n\nt:\t"+str(t))
    #################### < setting s1 values for next step     ####################
    train_data = set_new_s1_vals(train_data, spmn_t_structure, val_to_s_branch, s2_dict)
    ####################    setting s1 values for next step /> ####################
    new_s1s = train_data[:,t,0]
    train_data_unrolled = train_data[:,:t+1].reshape((-1,train_data.shape[2]))
    train_data_unrolled_old = train_data[:,:t].reshape((-1,train_data.shape[2]))
    train_data_unrolled_t = train_data[:,t].reshape((-1,train_data.shape[2]))
    # when horizon would go beyond last step, we reduce the horizon to learn the last few steps
    if t >= train_data_h.shape[1]:
        horizon -= 1
        train_data_h = get_horizon_train_data(data, horizon)
        partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = get_horizon_params(
                partialOrder, decNode, utilNode, scopeVars, meta_types, horizon
            )
        train_data_h[:,:,0] = train_data[:,:train_data_h.shape[1],0] # keeping train_data_h s1s updated
    train_data_h[:,t,0] = train_data[:,t,0]
    train_data_h_unrolled = train_data_h[:,t].reshape((-1,train_data_h.shape[2]))
    new_s1_vals = set(np.unique(new_s1s).astype(int))#i for i in range(s2_count)} - s1_vals
    new_s1_vals = new_s1_vals.difference(old_s1_vals) # only use truly new s1s for matching
    old_s1_vals = deepcopy(s1_vals)
    s1_vals = s1_vals.union(new_s1_vals)
    for new_val in new_s1_vals:
        likelihood_train_data = train_data_unrolled_t[train_data_unrolled_t[:,0]==new_val]
        if likelihood_train_data.shape[0] < 1: # TODO This shouldn't ever happen?
            print("\n\nfound 0 length new_val data slice for new_val == " + str(new_val))
            continue
        # adding s2s as nan to satisfy code
        nans_lh_data=np.empty((likelihood_train_data.shape[0],1))
        nans_lh_data[:] = np.nan
        likelihood_train_data = np.concatenate((likelihood_train_data,nans_lh_data),axis=1)
        # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
        likelihood_train_data[:,0] = np.nan
        print("\n\n< start matching for "+str(new_val)+" ...")
        # check if new s1 val is the same state as any existing states
        match_found = False
        for child in spmn_t_structure.children:
            #################### < matching to existing states    ####################
            if matches_state_branch(new_val, child, spmn_t_structure,
                    train_data, train_data_h, mean_branch_likelihoods,
                    use_chi2, chi2_threshold, likelihood_similarity_threshold,
                    s1_to_s2s, val_to_s_branch, t, horizon):
                match_found = True
                ####################    matching to existing states /> ####################
                #################### < adding new val to existing state    ####################
                # otherwise the new state is the same as this existing one,
                # so we add this value to the s1 node for that state-branch
                child_s1_vals = get_branch_s1_vals(child)
                child_s1_node = child.children[0]
                densities = child_s1_node.densities
                count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                for s1_val in child_s1_vals:
                    if s1_val == 0:
                        count_s1 = train_data.shape[0] # 1 starting state for each sequence
                    else:
                        count_s1 = s2_dict[s1_val].interface_links[child.children[0]]
                    densities[s1_val] = count_s1
                    count_child += count_s1
                count_new = likelihood_train_data.shape[0]
                count_child += count_new
                for s1_val in child_s1_vals:
                    densities[s1_val] /= count_child
                densities[new_val] = count_new / count_child
                child_s1_node.densities = densities
                # link s2 for new_val to this s1 node, or update counts
                if child_s1_node in s2_dict[new_val].interface_links:
                    s2_dict[new_val].interface_links[child_s1_node] += likelihood_train_data.shape[0]
                else:
                    s2_dict[new_val].interface_links[child_s1_node] = likelihood_train_data.shape[0]
                if new_val in val_to_s_branch:
                    val_to_s_branch[new_val] += [child]
                else:
                    val_to_s_branch[new_val] = [child]
                # as each branch is created to model a different distribution,
                #   we can expect that no further matches will be found.
                ####################    adding new val to existing state /> ####################
                break
        print("end matching for "+str(new_val)+" >")
        if not match_found: # if this new state represents a new distribution
            #################### < creating new branch for state    ####################
            # then create new child SPMN for the state
            print("\n\nnew state\t"+str(new_val)+"\n\n")
            new_spmn_data = train_data_h_unrolled[train_data_h_unrolled[:,0]==new_val]
            spmn_new_s1 = SPMN(
                    partialOrder_h,
                    decNode_h,
                    utilNode_h,
                    scopeVars_h,
                    meta_types_h,
                    cluster_by_curr_information_set=True,
                    util_to_bin = False
                )
            spmn_new_s1_structure = spmn_new_s1.learn_spmn(new_spmn_data, chi2_threshold)
            if horizon == 1:
                spmn_new_s1_structure, s2_count = assign_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            else:
                spmn_new_s1_structure, s2_count = replace_nextState_with_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            s2_dict[new_val].interface_links[spmn_new_s1_structure.children[0]] = new_spmn_data.shape[0]
            if new_val in val_to_s_branch:
                val_to_s_branch[new_val] += [spmn_new_s1_structure]
            else:
                val_to_s_branch[new_val] = [spmn_new_s1_structure]
            spmn_t_structure.children += [spmn_new_s1_structure]
            # update weights for each child SPMN
            weights = []
            for child in spmn_t_structure.children:
                child_s1_vals = get_branch_s1_vals(child)
                count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                for s1_val in child_s1_vals:
                    if s1_val == 0:
                        count_child += train_data.shape[0] # 1 starting state for each sequence
                    else:
                        count_child += s2_dict[s1_val].interface_links[child.children[0]]
                prob_child = count_child / train_data_unrolled.shape[0]
                weights.append(prob_child)
            normalized_weights = np.array(weights) / np.sum(weights)
            spmn_t_structure.weights = normalized_weights.tolist()
            spmn_t_structure = update_s_nodes(spmn_t_structure, len(scopeVars), s2_count)
            spmn_t_structure = assign_ids(spmn_t_structure)
            spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
            ####################    creating new branch for state /> ####################

spmn_t.spmn_structure = spmn_t_structure
rspmn = deepcopy(spmn_t)

file = open("data/"+str(dataset)+'/rspmn_'+'407'+'.pkle','wb')
import pickle
pickle.dump(rspmn, file)
file.close()

if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn_t_structure,  "plots/"+dataset+"/spmn_t.png", draw_interfaces=False)


def unroll_rspmn(rspmn_root, depth):
    #identify branches based on interface links
    root = deepcopy(rspmn_root)
    nodes = get_nodes_by_type(root)
    inteface_to_branch_dict = dict()
    for node in nodes:
        if type(node)==State and len(node.interface_links)==1 and\
            node.interface_links[0].id not in inteface_to_branch_dict:
            for child in root.children:
                # if the interface link leads to this child's s1 node
                if node.interface_links[0] == child.children[0]:
                    # then this s2 node leads to this branch
                    # -- the actual branch is the sibling of the s1 node
                    inteface_to_branch_dict[node.interface_links[0].id] = deepcopy(child.children[1])
                    break
    recursively_replace_s2_with_branch(root, inteface_to_branch_dict, depth-1)
    root = assign_ids(root)
    return root

def recursively_replace_s2_with_branch(root, inteface_to_branch_dict, remaining_depth):
    if remaining_depth == 0:
        return
    queue = [root]
    while len(queue) > 0:
        node = queue.pop(0)
        if type(node) is Product or type(node) is Sum or type(node) is Max:
            for i in range(len(node.children)):
                child = node.children[i]
                if type(child) is State and len(child.interface_links)==1:
                    node.children[i] = deepcopy(inteface_to_branch_dict[child.interface_links[0].id])
                    root = assign_ids(root)
                    root = rebuild_scopes_bottom_up(root)
                    recursively_replace_s2_with_branch(
                            node.children[i],
                            inteface_to_branch_dict,
                            remaining_depth-1
                        )
                elif type(child) is Product or type(child) is Sum or type(child) is Max:
                    queue.append(child)


unroll_len = 2
rspmn2 = unroll_rspmn(spmn_t_structure, unroll_len)

if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(rspmn2,  "plots/"+dataset+"/unroll"+str(unroll_len)+".png")

rspmn = deepcopy(spmn_t)

# test meu
from spn.algorithms.MEU import rmeu
input_data = np.array([0]+[np.nan]*4)
rmeu(rspmn, input_data, depth=2)

from spn.algorithms.MEU import meu
input_data = np.array([0]+[np.nan]*4)
meu(spmn0_structure, [input_data])

# load flspmn (reset caches)
file = open('rspmn_407.pkle','rb')
import pickle
flrspmn = pickle.load(file)
file.close()
rmeu(flrspmn, input_data, depth=6)

# inspect utilities
from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
nodes = get_nodes_by_type(rspmn.spmn_structure)
util_vals = [node.bin_repr_points for node in nodes if isinstance(node,Utility)]
state_nodes = [node for node in nodes if isinstance(node,State)]
max_nodes = [node for node in nodes if isinstance(node,Max)]

# tune weights with EM
# nans_em = np.empty((train_data_unrolled.shape[0],1))
# nans_em[:] = np.nan
# train_data_em = np.concatenate((train_data_unrolled,nans_em),axis=1)
# try:
#     EM_optimization(spmn_t_structure, train_data_em)
# except MemoryError:
#     error = True
#     partitions = 1
#     while(error):
#         partitions *= 5
#         error = False
#         start = 0
# for i in range(partitions):
#     if i < partitions-1:
#         end = start + train_data_em.shape[0] // partitions
#     else:
#         end = -1
#     try:
#         EM_optimization(spmn_t_structure, train_data_em[start:end], iterations=1)
#         start = end
#     except MemoryError:
#         error = True
#         break
