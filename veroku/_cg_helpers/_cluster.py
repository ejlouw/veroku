import numpy as np
import uuid

# TODO: add evidence observation functionality
# (perhaps only important for non-linear gaussian and similar approximate transformation factors)
from veroku.factors.gaussian import Gaussian
import copy
#from veroku.factors.nonlinear_gaussian import NonLinearGaussianMixture
#from veroku.factors.gaussian import split_gaussian

FIX_NON_PSD_MATRICES = False


class Cluster(object):
    """
    A class for instantiating clusters for use in ClusterGraph objects.
    """

    def __init__(self, factor, cluster_name_prefix=''):
        """
        Construct a cluster with an initial factor potential.
        :param factor: () The factor to use in the cluster
        :param name: The name of the cluster.
        """
        self._factor = factor
        # self._cluster_id = name if name is not None else str(uuid.uuid1())
        # self._cluster_id = name if name is not None else str(factor.var_names)
        self._cluster_id = cluster_name_prefix + ','.join(factor.var_names)
        self._neighbour_sepsets = {}  # neighbour_id as key
        self._received_message_factors = {}  # neighbour_id as key
        self._neighbour_references = []
        self._outward_message_paths = []
        self._special_evidence = None

    def add_special_evidence(self, special_evidence):
        self._special_evidence = special_evidence

    def remove_all_neighbours(self):
        self._neighbour_sepsets = {}  # neighbour_id as key

    def add_neighbour_if_appropriate(self, other_cluster):
        """
        Add a neighbour to this cluster's list of neighbours, if the sepset (overlap between the scope of this cluster
        and the other) length is greater than 0.
        :param other_cluster: (Cluster) The cluster to potentially add as a neighbour
        :return: (bool) Whether the other cluster was added as a neighbour or not.
        """
        this_cluster_var_set = set(self.var_names)
        sepset = list(this_cluster_var_set.intersection(set(other_cluster.var_names)))
        if len(sepset) > 0:
            self.add_neighbour(other_cluster)
            return True
        return False

    def add_neighbour(self, other_cluster, sepset=None):
        """
        Add a neighbour to this cluster's list of neighbours.
        :param other_cluster: (Cluster) The cluster to add as a neighbour
        :param sepset: The sepset between the neighbours (can be smaller than the scope intersection in order
                       to enforce the running intersection property)
        """
        this_cluster_var_set = set(self.var_names)
        if sepset is None:
            sepset = sorted(this_cluster_var_set.intersection(set(other_cluster.var_names)))
        assert len(sepset) > 0, 'Error: cant add neighbour with no overlapping variable scope.'
        self._neighbour_references.append(other_cluster)
        self._neighbour_sepsets[other_cluster.cluster_id] = list(sepset)

    #TODO: merge this with add_neighbour
    def add_outward_message_path(self, outward_message_path):
        self._outward_message_paths.append(outward_message_path)

    def make_message(self, neighbour_id):
        """
        Make a Message to send to the neighbour with a specified id.
        :param neighbour_id: The specified id corresponding to the neighbour cluster.
        :return: (Message) The Message
        """
        sepset_vars = self._neighbour_sepsets[neighbour_id]
        message_factor = self._factor.copy()
        if self._special_evidence is not None:
            evidence_vrs = list(self._special_evidence.keys())
            evidence_values = list(self._special_evidence.values())
            message_factor = message_factor.reduce(evidence_vrs, evidence_values)
        message_factor = message_factor.marginalize(sepset_vars, keep=True)
        if neighbour_id in self._received_message_factors:
            prev_received_message_factor = self._received_message_factors[neighbour_id]
            message_factor = message_factor.cancel(prev_received_message_factor)
            # TODO: remove this after TableFactor has been converted to LogTableFactor
            # message_factor = message_factor.normalize()
            if isinstance(message_factor, Gaussian) and FIX_NON_PSD_MATRICES:
                message_factor._fix_non_psd_matrices()
        message = Message(factor=message_factor, sender_id=self.cluster_id, receiver_id=neighbour_id)
        return message

    def receive_message(self, message):
        """
        Receive Message.
        :param message: (Message) The received Message
        """
        # Absorb message
        assert message.receiver_id == self.cluster_id, 'Error: Message not meant for this Cluster.'

        # EXPERIMENTAL START
        #if isinstance(self._factor, NonLinearGaussianMixture) and isinstance(message.factor, Gaussian) and len(message.factor.var_names) == 1:
        #    message_factor = split_gaussian(message.factor)
        #    self._factor = self._factor.multiply(message_factor)
        # EXPERIMENTAL END
        #else:
        self._factor = self._factor.multiply(message.factor)

        # Cancel out any message previously received from sender cluster
        sender_id = message.sender_id
        if sender_id in self._received_message_factors:
            prev_received_message_factor = self._received_message_factors[sender_id]
            self._factor = self._factor.cancel(prev_received_message_factor)
        # TODO: remove this after TableFactor has been converted to LogTableFactor
        #self._factor = self._factor.normalize()
        self._received_message_factors[message.sender_id] = message.factor.copy()

    def get_sepset(self, neighbour_id):
        """
        Get the sepset between this cluster and another.
        :param neighbour_id: The id of the other cluster.
        :return: (list) The sepset
        """
        sepset = []
        try:
            sepset = self._neighbour_sepsets[neighbour_id]
        except KeyError:
            pass
        return sepset

    @property
    def neighbour_ids(self):
        """
        Get a list of ids corresponding to the cluster's neigbours.
        """
        return list(self._neighbour_sepsets.keys())

    @property
    def var_names(self):
        """
        The variable names associated with the cluster factor.
        :return: the var_names
        """
        return self._factor.var_names

    @property
    def cluster_id(self):
        """
        This cluster's id.
        :return: The cluster_id
        """
        return self._cluster_id


class Message(object):

    """
    A class for instantiating Message objects.
    """

    def __init__(self, factor, sender_id, receiver_id):
        """
        The constructor.
        :param factor: The Message factor.
        :param sender_id:  The cluster_id of the sender cluster
        :param receiver_id: The cluster_id of the receiver cluster
        """
        self.uuid = str(uuid.uuid1())
        self._sender_id = sender_id
        self._receiver_id = receiver_id
        self._factor = factor

    def equals(self, other, rtol=1e-05, atol=1e-05):
        if self._sender_id != other.sender_id:
            return False
        if self._receiver_id != other._receiver_id:
            return False
        if not self._factor.equals(other._factor, rtol=rtol, atol=atol):
            return False
        return True

    def __repr__(self):
        repr_str = 'sender_id: ' + self._sender_id + '\n' + \
                   'receiver_id: ' + self._receiver_id + '\n' + \
                   'factor: \n' + self._factor.__repr__()
        return repr_str

    @property
    def sender_id(self):
        """
        This Message's sender id
        :return: The sender_id
        """
        return self._sender_id

    @property
    def receiver_id(self):
        """
        This Message's receiver id
        :return: The receiver_id
        """
        return self._receiver_id

    @property
    def factor(self):
        """
        This Message's factor
        :return: The factor
        """
        return self._factor.copy()

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between the message factor and a vacuous version of it.
        :return: The KL-divergence
        """
        return self.factor.distance_from_vacuous()

    def kl_divergence(self, message):
        """
        Get the KL-divergence D_KL(self.factor || message.factor) between a this message factor and another message factor.
        :param message: The other message of which the factor will be used to compare to this message's factor.
        :return: The KL-divergence
        """
        distance = self.factor.kl_divergence(message.factor)
        return distance

    @property
    def var_names(self):
        """
        The variable names associated with the cluster factor.
        :return: the var_names
        """
        return self._factor.var_names

    def copy(self):
        return Message(self.factor.copy(), copy.deepcopy(self.sender_id), copy.deepcopy(self.receiver_id))