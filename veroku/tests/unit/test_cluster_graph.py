
import unittest

from veroku.cluster_graph import ClusterGraph
from veroku._cg_helpers._cluster import Message
from veroku.factors.gaussian import Gaussian
import numpy as np


class TestClusterGraph(unittest.TestCase):
    """
    A test class for the ClusterGraph class
    """

    def test_correct_message_passing(self):
        """
        Check that the correct messages are passed in the correct order.
        """
        fa = Gaussian(var_names=['a'], cov=[[0.5]], mean=[0.0], log_weight=3.0)

        fab = Gaussian(var_names=['a', 'b'], cov=[[10, 9], [9, 10]], mean=[0, 0], log_weight=0.0)
        fabfa = fab.absorb(fa)

        fac = Gaussian(var_names=['a', 'c'], cov=[[10, 3], [3, 10]], mean=[0, 0], log_weight=0.0)
        fbd = Gaussian(var_names=['b', 'd'], cov=[[15, 4], [4, 15]], mean=[0, 0], log_weight=0.0)

        cg = ClusterGraph([fa, fab, fac, fbd])

        # expected messages

        # from cluster 0 (fabfa) to cluster 1 (fac)
        msg_1_factor = fabfa.marginalize(vrs=['a'], keep=True)
        msg_1 = Message(msg_1_factor, 'c0#a,b', 'c1#a,c')

        # from cluster 0 (fabfa) to cluster 2 (fbd)
        msg_2_factor = fabfa.marginalize(vrs=['b'], keep=True)
        msg_2 = Message(msg_2_factor, 'c0#a,b', 'c2#b,d')

        # from cluster 1 (fac) to cluster 0 (fabfa)
        msg_3_factor = fac.marginalize(vrs=['a'], keep=True)
        msg_3 = Message(msg_3_factor, 'c1#a,c', 'c0#a,b')

        # from cluster 2 (fbd) to cluster 0 (fabfa)
        msg_4_factor = fbd.marginalize(vrs=['b'], keep=True)
        msg_4 = Message(msg_4_factor, 'c2#b,d', 'c0#a,b')

        expected_messages = [msg_1, msg_2, msg_3, msg_4]

        # Test that the factors of the cluster in the cluster graph are correct
        expected_cluster_factors = [fabfa.copy(), fac.copy(), fbd.copy()]
        actual_cluster_factors = [c._factor for c in cg._clusters]
        key_func = lambda f: ''.join(f.var_names)
        actual_cluster_factors = sorted(actual_cluster_factors, key=key_func)
        expected_cluster_factors = sorted(expected_cluster_factors, key=key_func)

        for actual_f, expect_f in zip(actual_cluster_factors, expected_cluster_factors):
            self.assertEqual(actual_f, expect_f)

        # See note below
        for gmp in cg.graph_message_paths:
            receiver_cluster_id = gmp.receiver_cluster._cluster_id
            sender_cluster_id = gmp.sender_cluster._cluster_id
            message_vars = gmp.sender_cluster.get_sepset(receiver_cluster_id)
            dim = len(message_vars)
            almost_vacuous = Gaussian(var_names=message_vars,
                                      cov=np.eye(dim)*1e10, mean=np.zeros([dim,1]), log_weight=0.0)
            gmp.previously_sent_message = Message(sender_id=sender_cluster_id,
                                                  receiver_id=receiver_cluster_id, factor=almost_vacuous)
            gmp.update_next_information_gain()

        cg.process_graph(tol=0, max_iter=1, debug=True)

        # Note
        # Now we want to ensure and check a certain message order. The problem is that if more than one KLD is inf,
        # there is no correct sorting order. This potentially points to a trade-off between easy 'distance from vacuous'
        # calculations at the start of message passing (and not ensuring that the most informative message is sent) and
        # maybe rather calculating a distance from almost vacuous and ensuring that the most informative messages are
        # sent first. Infinities might not be sortable, but that does not mean they are equal.

        actual_messages = cg.messages_passed
        for actual_message, expected_message in zip(actual_messages, expected_messages):
            self.assertEqual(actual_message.sender_id, expected_message.sender_id)
            self.assertEqual(actual_message.receiver_id, expected_message.receiver_id)
            self.assertTrue(actual_message.equals(expected_message, rtol=1e-03, atol=1e-03))

    def test_correct_posterior_marginal_weights(self):
        """
        Check that the marginal weights are correct.
        """
        fa = Gaussian(var_names=['a'], cov=[[0.5]], mean=[0.0], log_weight=3.0)
        fab = Gaussian(var_names=['a', 'b'], cov=[[10, 9], [9, 10]], mean=[0, 0], log_weight=0.0)
        fac = Gaussian(var_names=['a', 'c'], cov=[[10, 3], [3, 10]], mean=[0, 0], log_weight=0.0)
        fbd = Gaussian(var_names=['b', 'd'], cov=[[15, 4], [4, 15]], mean=[0, 0], log_weight=0.0)
        factors = [fa, fab, fac, fbd]
        cg = ClusterGraph(factors)
        # TODO: see why this fails with max_iter=1
        cg.process_graph(tol=0, max_iter=2, debug=True)
        # check posterior weight
        joint = factors[0]
        for f in factors[1:]:
            joint = joint.absorb(f)
        expected_log_weight = joint.get_log_weight()

        # the marginals are all marginals of the same distribution and should therefore have the same weight
        # (the integrand is the same, regardless of the order in which the variables are integrated out)
        actual_log_weights = [cluster._factor.get_log_weight() for cluster in cg._clusters]
        self.assertTrue(np.allclose(actual_log_weights, expected_log_weight))
