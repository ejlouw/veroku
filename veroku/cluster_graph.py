from IPython.core.display import Image, display
from graphviz import Graph
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from graphviz import Source

from veroku._cg_helpers._cluster import Cluster
import veroku._cg_helpers._animation as cg_animation
from veroku.factors._factor_utils import get_subset_evidence
import matplotlib.pyplot as plt
import collections

# TODO: optimise _pass_message
# TODO: improve sepsets selection for less loopiness
# TODO: Optimisation: messages from clusters that did not receive any new messages in the previous round, do not need
#  new messages calculated.


def sort_almost_sorted(a_deque, key_func):
    """
    Sort a deque like that where only the first element is potentially unsorted
    and should probably be last and the rest of the deque is sorted in descending order.
    """
    a_deque.append(a_deque.popleft())
    if key_func(a_deque[-2]) <= key_func(a_deque[-1]) :
        return
    else:
        print('key_func(a_deque[-1]) = ', key_func(a_deque[-1]))
        print('key_func(a_deque[-2]) = ', key_func(a_deque[-2]))
        print()
        raise NotImplementedError('not implemented, add efficient sorting here.')


def _evidence_reduce_factors(factors, evidence):
    """
    Observe relevant evidence for each factor.
    :param factors:
    :param evidence:
    :return:
    """
    reduced_factors = []
    for i, factor in enumerate(factors):
        if evidence is not None:
            vrs, values = get_subset_evidence(all_evidence_dict=evidence,
                                              subset_vars=factor.var_names)
            if len(vrs) > 0:
                factor = factor.reduce(vrs, values)
        reduced_factors.append(factor.copy())
    return reduced_factors


def make_factor_name(factor):
    return str(factor.var_names).replace("'", '')


def _absorb_subset_factors(factors):
    """
    Absorb any factors that has a scope that is a subset of another factor into such a factor.
    :param factors:
    :return:
    """
    factors_absorbtion_dict = {i: [] for i in range(len(factors))}
    final_graph_cluster_factors = []
    # factors: possibly smaller list of factors after factors which have a scope that is a subset of another factor have
    # been absorbed by the larger one.
    factor_processed_mask = [0] * len(factors)
    for i, factor_i in enumerate(factors):
        if not factor_processed_mask[i]:
            factor_product = factor_i.copy()
            for j, factor_j in enumerate(factors):
                if i != j:
                    if not factor_processed_mask[j]:
                        if set(factor_j.var_names) < set(factor_product.var_names):
                            try:
                                factor_product = factor_product.multiply(factor_j)
                                factors_absorbtion_dict[i].append(j)
                                factor_processed_mask[j] = 1
                                factor_processed_mask[i] = 1
                            except NotImplementedError:
                                print(f'Warning: could not multiply {type(factor_product)} with {type(factor_j)} (Not Implemented)')
            if factor_processed_mask[i]:
                final_graph_cluster_factors.append(factor_product)
    for i, factor_i in enumerate(factors):  # add remaining factors
        if not factor_processed_mask[i]:
            factor_processed_mask[i] = 1
            final_graph_cluster_factors.append(factor_i)
    assert all(factor_processed_mask), 'Error: Some factors where not included during variable subset processing.'
    return final_graph_cluster_factors


def _make_subset_factor_df(subset_dict):
    """
    Make a ...
    :param subset_dict: (dict) A dictionary mapping factors to factors that have subset scopes
    :return:
    """
    keys = list(subset_dict.keys())
    values = list(subset_dict.values())
    assert(len(keys) == len(values))
    # TODO: This raises different list lengths warning (see below). Investigate this.
    #   VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences...
    data = np.array([keys, values]).T
    df = pd.DataFrame(columns=['factor_index', 'subfactor_indices'],
                      data=data)
    df['num_subfactors'] = df['subfactor_indices'].apply(lambda x: len(x))
    df.sort_values(by='num_subfactors', inplace=True, ascending=False)
    return df


class ClusterGraph(object):
    
    def __init__(self, factors, evidence=None, special_evidence=dict(),
                 make_animation_gif=False, disable_tqdm=False, verbose=False):
        """
        Construct a Cluster graph from a list of factors
        :param factors: (list of factors) The factors to construct the graph from
        :param evidence: (dict) evidence dictionary (mapping variable names to values) that should be used to reduce
            factors before building the cluster graph.
        :param special_evidence: (dict) evidence dictionary (mapping variable names to values) that should be used in
            the calculation of messages, and not to reduce factors. This allows factor approximations - such as the
            non-linear Gaussian to be iteratively refined.
        """

        self.num_messages_passed = 0
        self.make_animation_gif = make_animation_gif
        self.special_evidence = special_evidence
        self.disable_tqdm = disable_tqdm
        self.last_passed_message_factors_dict = dict()
        self.verbose = verbose
        # new
        self.last_sent_message_dict = {}  # {(rec_cluster_id1, rec_cluster_id1): msg1, ...}

        self.sync_message_passing_max_distances = None
        self.sync_message_passing_max_distances = None

        all_evidence_vars = set(self.special_evidence.keys())
        if evidence is not None:
            evidence_vars = set(evidence.keys())
            all_evidence_vars = all_evidence_vars.union(evidence_vars)
        all_factors_copy = _evidence_reduce_factors(factors, evidence)
        final_graph_cluster_factors = _absorb_subset_factors(all_factors_copy)

        clusters = [Cluster(factor, cluster_name_prefix=f'c{i}#') for i, factor in
                    enumerate(final_graph_cluster_factors)]

        self._set_non_rip_sepsets_dict(clusters, all_evidence_vars)
        self._clusters = clusters

        # Add special evidence to factors
        for cluster in self._clusters:
            cluster_special_evidence_vars, cluster_special_evidence_values = get_subset_evidence(self.special_evidence, cluster.var_names)
            print('cluster_special_evidence_vars = ', cluster_special_evidence_vars)
            print('cluster_special_evidence_values = ', cluster_special_evidence_values)
            cluster_special_evidence = dict(zip(cluster_special_evidence_vars, cluster_special_evidence_values))
            cluster.add_special_evidence(cluster_special_evidence)

        self.graph_message_paths = collections.deque([])
        self._build_graph()

        # TODO: consolidate these two, if possible
        self.message_passing_log_df = None
        self.message_passing_animation_frames = []

    def _set_non_rip_sepsets_dict(self, clusters, all_evidence_vars):
        """
        Calculate the preliminary sepsets dict before the RIP property is enforced.
        :param clusters:
        :param all_evidence_vars:
        :return:
        """
        self._non_rip_sepsets = {}
        for i in tqdm(range(len(clusters)), disable=self.disable_tqdm):
            vars_i = clusters[i].var_names
            for j in range(i + 1, len(clusters)):
                vars_j = clusters[j].var_names
                sepset = set(vars_j).intersection(set(vars_i)) - all_evidence_vars
                self._non_rip_sepsets[(i, j)] = sepset
                self._non_rip_sepsets[(j, i)] = sepset

    def _build_graph(self):
        """
        Add the cluster sepsets, graphviz graph and animation graph (for message_passing visualisation).
        """
        # Check for non-unique cluster_ids (This should never be the case)
        cluster_ids = [cluster.cluster_id for cluster in self._clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError(f'Non-unique cluster ids: {cluster_ids}')

        self._conditional_print('Info: Building graph.')
        self._graph = Graph(format='png')

        rip_sepsets_dict = self._get_running_intersection_sepsets()

        self._conditional_print(f'Debug: number of clusters: {len(self._clusters)}')
        for i in tqdm(range(len(self._clusters)), disable=self.disable_tqdm):

            node_i_name = self._clusters[i]._cluster_id
            self._graph.node(name=node_i_name, label=node_i_name, style='filled', fillcolor='white', color='black')
            for j in range(i + 1, len(self._clusters)):

                if (i, j) in rip_sepsets_dict:
                    sepset = rip_sepsets_dict[(i, j)]
                    assert len(sepset) > 0, 'Error: empty sepset'
                    self._clusters[i].add_neighbour(self._clusters[j], sepset=sepset)
                    self._clusters[j].add_neighbour(self._clusters[i], sepset=sepset)

                    gmp_ij = GraphMessagePath(self._clusters[i], self._clusters[j])
                    gmp_ji = GraphMessagePath(self._clusters[j], self._clusters[i])
                    self.graph_message_paths.append(gmp_ij)
                    self.graph_message_paths.append(gmp_ji)
                    self._clusters[i].add_outward_message_path(gmp_ij)
                    self._clusters[j].add_outward_message_path(gmp_ji)

                    # Graph animation
                    node_j_name = self._clusters[j]._cluster_id
                    sepset_node_label = ','.join(sepset)
                    sepset_node_name = cg_animation.make_sepset_node_name(node_i_name, node_j_name)
                    self._graph.node(name=sepset_node_name, label=sepset_node_label, shape='rectangle')
                    self._graph.edge(node_i_name, sepset_node_name, color='black', penwidth='2.0')
                    self._graph.edge(sepset_node_name, node_j_name, color='black', penwidth='2.0')
        print('num self.graph_message_paths: ', len(self.graph_message_paths))

    def _conditional_print(self, message):
        if self.verbose:
            print(message)

    def plot_message_convergence(self, old=False):
        """
        Plot the the KL-divergence between the messages and their previous instances to indicate the message passing
        convergence.
        :param old: Whether or not to use the older function for synchronous message passing convergence.
        :type old: Bool
        """
        if self.sync_message_passing_max_distances:
            assert self.message_passing_log_df is None, 'Error: it seems bot sync and async message passing was run.'
            plt.plot(self.sync_message_passing_max_distances)
        else:
            if old:
                self._plot_message_convergence_old()
            else:
                self._plot_message_convergence_new()

    def _plot_message_convergence_old(self):
        """
        Plot the message passing convergence over the consecutive iterations.
        """
        # TODO: combine with one below - this one gets messed up by inf values
        message_kld_df = self.message_passing_log_df
        columns_mask = message_kld_df.columns.to_series().str.contains('distance_from_previous')
        message_kld_distances_only_df = message_kld_df[message_kld_df.columns[columns_mask]]
        message_kld_distances_only_df.columns = range(message_kld_distances_only_df.shape[1])

        title = 'KL Divegences between Messages and Previous Iteration Messages'
        ax = message_kld_distances_only_df.plot.box(figsize=[20, 10], title=title)
        ax.title.set_size(20)
        ax.set_ylabel('KLD', fontsize=15)
        ax.set_xlabel('message passing iteration', fontsize=15)

    def _plot_message_convergence_new(self, figsize=[10, 5]):
        # TODO: improve this (inf value workaround is a bit hacky)
        df = self.message_passing_log_df
        kl_cols = [col for col in df.columns if 'distance' in col]
        kl_df = df[kl_cols]

        no_inf_df = kl_df.replace([np.inf], 0) * 2
        max_no_inf = np.max(no_inf_df.values)

        no_inf_df = kl_df.replace([-np.inf], 0) * 2
        min_no_inf = np.min(no_inf_df.values)

        inf_to_max_df = kl_df.replace([np.inf], max_no_inf * 2)
        inf_to_max_min_df = inf_to_max_df.replace([-np.inf], min_no_inf * 2)
        data = inf_to_max_min_df.values
        max_kl_div_per_iteration = np.max(data, axis=0)
        plt.figure(figsize=figsize)
        plt.plot(np.log(max_kl_div_per_iteration))
        plt.title('Message Passing Convergence', fontsize=15)
        plt.ylabel('log max message kld')
        plt.xlabel('message passing iteration')

    def _get_unique_vars(self):
        all_vars = []
        for cluster in self._clusters:
            all_vars += (cluster.var_names)
        unique_vars = list(set(all_vars))
        return unique_vars

    def _get_vars_min_spanning_trees(self):
        all_vars = self._get_unique_vars()
        var_graphs = {var: nx.Graph() for var in all_vars}
        num_clusters = len(self._clusters)
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                sepset = self._non_rip_sepsets[(i, j)]
                for var in sepset:
                    var_graphs[var].add_edge(i, j, weight=1)
        var_spanning_trees = dict()
        for var in all_vars:
            var_spanning_trees[var] = nx.minimum_spanning_tree(var_graphs[var])
        return var_spanning_trees

    def _get_running_intersection_sepsets(self):
        edge_sepset_dict = {}
        unique_vars = self._get_unique_vars()
        min_span_trees = self._get_vars_min_spanning_trees()
        self._conditional_print("Info: Getting unique variable spanning trees.")
        for i in tqdm(range(len(unique_vars)), disable=self.disable_tqdm):
            var = unique_vars[i]
            min_span_tree = min_span_trees[var]
            for edge in min_span_tree.edges():
                if edge in edge_sepset_dict:
                    edge_sepset_dict[edge].append(var)
                else:
                    edge_sepset_dict[edge] = [var]
        return edge_sepset_dict

    def show(self):
        """
        Show the cluster graph.
        """
        self._graph.render('/tmp/test.gv', view=False)
        image = Image('/tmp/test.gv.png')
        display(image)

    def save_graph_image(self, filename):
        """
        Save image of the graph.
        :param filename: The filename of the file.
        """
        # Source(self._graph, filename="/tmp/test.gv", format="png")
        Source(self._graph, filename=filename, format="png")

    def get_marginal(self, vrs):
        """
        Search the graph for a specific variable and get that variables marginal (posterior marginal if process_graph
        has been run previously).
        :return: The marginal
        """
        for cluster in self._clusters:
            if set(vrs) <= set(cluster.var_names):
                factor = cluster._factor.copy()
                evidence_vrs, evidence_values = get_subset_evidence(self.special_evidence, factor.var_names)
                if len(evidence_vrs) > 0:
                    factor = factor.reduce(evidence_vrs, evidence_values)
                marginal = factor.marginalize(vrs, keep=True)
                return marginal
        raise ValueError(f'No cluster with variables containing {vrs}')

    def get_posterior_joint(self):
        """
        Get the posterior joint distribution.
        """
        # TODO: add functionality for efficiently getting a posterior marginal over any subset of variables and replace
        #  the get_marginal function above.
        cluster_product = self._clusters[0]._factor.joint_distribution
        for cluster in self._clusters[1:]:
            cluster_product = cluster_product.multiply(cluster._factor.joint_distribution)
        last_passed_message_factors = list(self.last_passed_message_factors_dict.values())
        if len(last_passed_message_factors) == 0:
            assert self.num_messages_passed == 0
            return cluster_product
        message_product = last_passed_message_factors[0]
        for message_factor in last_passed_message_factors[1:]:
            message_product = message_product.multiply(message_factor)
        joint = cluster_product.cancel(message_product)
        return joint

    def process_graph(self, tol=1e-3, max_iter=50):
        """
        Process the graph by performing message passing until convergence.
        """
        self._process_graph_sync(tol=tol, max_iter=max_iter)

    # New Synchronous version
    #  TODO: make this more efficient, if no messages have been received by a cluster in the previous round, the next
    #        message iterations from that cluster will be the same.
    def _process_graph_sync(self, tol, max_iter):
        """
        Perform synchronous message passing until convergence (or maximum iterations).
        """
        self.sync_message_passing_max_distances = []
        if len(self._clusters) == 1:
            # The Cluster Graph contains only single cluster. Message passing not possible or necessary.
            self._clusters[0]._factor = self._clusters[0]._factor.reduce(vrs=self.special_evidence.keys(),
                                                                         values=self.special_evidence.values())
            return

        # TODO: see if the definition of max_iter can be improved
        key_func = lambda x: x.next_information_gain
        max_message_passes = max_iter*len(self.graph_message_paths)
        #TODO: find a better solution than converting back to deque
        self.graph_message_paths = collections.deque(sorted(self.graph_message_paths, key=key_func, reverse=True))
        print('[gmp.next_information_gain for gmp in graph_message_paths]: \n', [gmp.next_information_gain for gmp in self.graph_message_paths])
        for i in range(max_message_passes):
            self.graph_message_paths[0].pass_next_message()
            sort_almost_sorted(self.graph_message_paths, key_func=key_func)
            print(i)
            if self.graph_message_paths[0].next_information_gain < tol:
                print(f'{self.graph_message_paths[0].next_information_gain} < tol')
                return

        #if self.make_animation_gif:
        #    cg_animation.add_message_pass_animation_frames(graph=self._graph,
        #                                                   frames=self.message_passing_animation_frames,
        #                                                   node_a_name=message.sender_id,
        #                                                   node_b_name=receiver_cluster_id)

    def _make_message_passing_animation_gif(self):
        print('Making message passing animation.')
        self.message_passing_animation_frames[0].save(fp='./graph_animation.gif',
                                                      format='GIF',
                                                      append_images=self.message_passing_animation_frames[1:],
                                                      save_all=True, duration=400, loop=0)

    @property
    def graph(self):
        return self._graph


class GraphMessagePath:
    """
    A specific path (direction along an edge) in a graph along which a message can be passed.
    """

    def __init__(self, sender_cluster, receiver_cluster):
        self.sender_cluster = sender_cluster
        self.receiver_cluster = receiver_cluster
        self.previous_message = None
        self.next_message = self.sender_cluster.make_message(self.receiver_cluster._cluster_id)
        self.update_next_information_gain()

    def update_next_information_gain(self):
        if self.previous_message is None:
            self.next_information_gain = self.next_message.distance_from_vacuous()
        else:
            # "In the context of machine learning, KL(P||Q) is often called the information gain achieved if Q is
            # used instead of P." - wikipedia
            # We typically want to know which new message (Q) will result in the largest information gain if it replaces
            # the message (P)
            # message: previous_message (P)
            # factor: next message (Q)
            self.next_information_gain = self.next_message.kl_divergence(self.previous_message)

    def recompute_next_message(self):
        """
        Recompute the next message

        """
        new_next_message = self.sender_cluster.make_message(self.receiver_cluster._cluster_id)
        self.previous_message, self.next_message = self.next_message, new_next_message
        self.update_next_information_gain()

    def pass_next_message(self):
        self.receiver_cluster.receive_message(self.next_message)
        for gmp in self.receiver_cluster._outward_message_paths:
            gmp.recompute_next_message()