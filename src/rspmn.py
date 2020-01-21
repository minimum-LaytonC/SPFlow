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

## using rdc to find correlation
# ds_context = get_ds_context(train_data_unrolled, [0,1,2,3], spmn_t.params)
# split_cols = get_split_cols_RDC_py()
# data_slices = split_cols(train_data_s1_selected, ds_context, scope)
# correlated = True
# for cluster, cluster_scope, weight in data_slices:
#     if 0 in cluster_scope and len(cluster_scope)==1:
#         correlated = False
#
# correlated
##############################

dataset = "repeated marbles"

if dataset == "repeated marbles":
    df = pd.DataFrame.from_csv("repeated_marbles_10000x20.tsv", sep='\t')
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
    horizon = 2
elif dataset == "tiger":
    df = pd.DataFrame.from_csv("tiger_noisy_obs_100000x4.tsv", sep='\t')
    data = df.values.reshape(100000,4,3)
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
    horizon = 2

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
    decNode_h = decNode+[decNode[0]+"_t+"+str(i) for i in range (1,horizon)]
    utilNode_h = utilNode
    scopeVars_h = scopeVars + [var+"_t+"+str(i) for var in scopeVars[1:] for i in range (1,horizon)]
    meta_types_h = meta_types+[MetaType.DISCRETE]*(len(scopeVars_h)-len(meta_types))
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
spmn0_structure = spmn0.learn_spmn(train_data_h[:,0])

s2_dict = {}

import queue
def replace_nextState_with_s2(spmn,s2_scope_idx,s2_count=1, s2_dict=s2_dict):
    scope_t1 = {i for i in range(s2_scope_idx)}
    q = q = queue.Queue()
    q.put(spmn)
    while not q.empty():
        node = q.get()
        if isinstance(node, Product):
            for child in node.children:
                # if the child has no variables from the first timestep
                if len(set(child.scope) & scope_t1) == 0:
                    # then replace it with an s2 node
                    node.children.remove(child)
                    new_s2 = State(
                            [s2_count,s2_count+1],
                            [1],
                            [s2_count],
                            scope=s2_scope_idx
                        )
                    node.children.append(new_s2)
                    s2_dict[s2_count] = new_s2
                    s2_count += 1
                    break
                else:
                    q.put(child)
        elif isinstance(node, Max) or isinstance(node, Sum):
            for child in node.children:
                q.put(child)
    return spmn, s2_count

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
                node.children.append(new_s2)
                s2_count += 1
            else:
                for child in node.children:
                    q.put(child)
    return spmn, s2_count

from spn.io.Graphics import plot_spn
if dataset == "repeated marbles":
    plot_spn(spmn0_structure, "repeated_marbles_spmn0_h.png")
elif dataset == "tiger":
    plot_spn(spmn0_structure, "tiger_spmn0_NEWEST.png")

def update_s_nodes(spmn,s2_scope_idx,s2_count):
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
if dataset == "repeated marbles":
    plot_spn(spmn0_structure, "repeated_marbles_spmn0_with_s2_h.png")
elif dataset == "tiger":
    plot_spn(spmn0_structure, "tiger_spmn0_with_s2_h_NEWEST.png")


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

# learn new sub spmn branches based on state values
s1_vals = {0}
chi2_thresh = 0.05
from spn.algorithms.MPE import mpe
from sklearn.feature_selection import chi2
for t in range(1, train_data.shape[1]):
    # s1 at t is s2 at t-1
    train_data_s2 = np.concatenate((train_data,nans),axis=2)
    train_data[:,t,0] = mpe(spmn_t_structure, train_data_s2[:,t-1])[:,len(scopeVars)]
    train_data_unrolled = train_data[:,:t+1].reshape((-1,train_data.shape[2]))
    if t >= train_data_h.shape[1]:
        horizon -= 1
        train_data_h = get_horizon_train_data(data, horizon)
        partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = get_horizon_params(
                partialOrder, decNode, utilNode, scopeVars, meta_types, horizon
            )
        train_data_h[:,:,0] = train_data[:,:train_data_h.shape[1],0]
    train_data_h[:,t,0] = train_data[:,t,0]
    train_data_h_unrolled = train_data_h[:,:t+1].reshape((-1,train_data_h.shape[2]))
    new_s1_vals = {i for i in range(s2_count)} - s1_vals
    s1_vals = s1_vals.union(new_s1_vals)
    for new_val in new_s1_vals:
        # check if new s1 val is the same state as any existing states
        matchFound = False
        for child in spmn_t_structure.children:
            s1_node_vals = np.array(child.children[0].bin_repr_points)
            s1_node_nonzero = np.ceil(child.children[0].densities).astype(bool)
            # child_s1_vals is the set of states this branch represents
            child_s1_vals = list(s1_node_vals[s1_node_nonzero].astype(int))
            testing_s1_vals = child_s1_vals + [new_val]
            mask = np.isin(train_data_unrolled[:,0],testing_s1_vals)
            # select data corresponding to this node's states and the new state
            train_data_s1_selected = train_data_unrolled[mask]
            # look for correlations between state and the other values
            # TODO: change to RDC
            print("\nstate, child_s1_vals:\t"+str(new_val)+",\t"+str(child_s1_vals))
            min_chi2_pvalue = np.min(chi2(
                    np.abs(np.delete(train_data_s1_selected,0,axis=1)),
                    train_data_s1_selected[:,0]
                )[1])
            print("min_chi2_pvalue:\n" + str(min_chi2_pvalue))
            if min_chi2_pvalue < chi2_thresh:
                # if s1 is correlated with any other variables, then the
                # then the new value is a functionally different state
                continue
            # otherwise the new state is the same as this existing one; add it to the s1 node
            else:
                child_s1_node = child.children[0]
                densities = child_s1_node.densities
                for s1_val in child_s1_vals:
                    s1_count = np.sum(train_data_s1_selected[:,0]==s1_val)
                    s1_prob = s1_count / train_data_s1_selected.shape[0]
                    densities[s1_val] = s1_prob
                count_new = np.sum(train_data_s1_selected[:,0]==new_val)
                s1_prob_new = count_new / train_data_s1_selected.shape[0]
                densities[new_val] = s1_prob_new
                child_s1_node.densities = densities
                # update weights in child using new data
                nans_em = np.empty((train_data_s1_selected.shape[0],1))
                nans_em[:] = np.nan
                train_data_em = np.concatenate((train_data_s1_selected,nans_em),axis=1)
                child = assign_ids(child)
                child = rebuild_scopes_bottom_up(child)
                EM_optimization(child, train_data_em)
                spmn_t_structure = assign_ids(spmn_t_structure)
                spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
                # link s2 for new_val to this s1 node
                s2_dict[new_val].interface_links.append(child_s1_node)
                matchFound = True
                break
        if not matchFound: # if this new state is functionally different than all existing states,
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
            spmn_new_s1_structure = spmn_new_s1.learn_spmn(new_spmn_data)
            if horizon == 1:
                spmn_new_s1_structure, s2_count = assign_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            else:
                spmn_new_s1_structure, s2_count = replace_nextState_with_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            s2_dict[new_val].interface_links.append(spmn_new_s1_structure.children[0])
            spmn_t_structure.children += [spmn_new_s1_structure]
            # update weights for each child SPMN
            weights = []
            for child in spmn_t_structure.children:
                s1_node_vals = np.array(child.children[0].bin_repr_points)
                s1_node_nonzero = np.ceil(child.children[0].densities).astype(bool)
                # child_s1_vals is the set of states this branch represents
                child_s1_vals = list(s1_node_vals[s1_node_nonzero].astype(int))
                count_child = np.sum(np.isin(train_data_h_unrolled[:,0],child_s1_vals))
                prob_child = count_child / train_data_h_unrolled.shape[0]
                weights.append(prob_child)
            normalized_weights = np.array(weights) / np.sum(weights)
            spmn_t_structure.weights = normalized_weights.tolist()
            spmn_t_structure = update_s_nodes(spmn_t_structure, len(scopeVars), s2_count)
            spmn_t_structure = assign_ids(spmn_t_structure)
            spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)

from spn.io.Graphics import plot_spn
if dataset == "repeated marbles":
    plot_spn(spmn_t_structure, "repeated_marbles_spmn_t_h.png", draw_interfaces=True)
elif dataset == "tiger":
    plot_spn(spmn_t_structure, "tiger_spmn_t_h_NEWEST.png", draw_interfaces=True)

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
    # TODO fix scope stuff for MEU ?.
    # or maybe it'll be easier to just write a new MEU function with a specified depth...
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

rspmn2 = unroll_rspmn(spmn_t_structure, 2)

if dataset == "repeated marbles":
    plot_spn(rspmn2, "repeated_marbles_unroll2.png")
elif dataset == "tiger":
    plot_spn(rspmn2, "tiger__unroll2.png")

from spn.algorithms.MEU import meu
# meu(spmn_t_structure, np.array([[0,1,np.nan,np.nan,np.nan]]))
