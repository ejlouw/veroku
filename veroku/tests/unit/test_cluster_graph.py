
import unittest

from veroku.cluster_graph import ClusterGraph
from veroku._cg_helpers._cluster import Message
from veroku.factors.gaussian import Gaussian


class TestClusterGraph(unittest.TestCase):
    """
    A test class for the ClusterGraph class
    """

    def test_correct_message_passing(self):
        fa = Gaussian(var_names=['a'], cov=[[1]], mean=[7.0], log_weight=0.0)
        fab = Gaussian(var_names=['a', 'b'], cov=[[10, 2], [2, 10]], mean=[2, 3], log_weight=0.0)
        fabfa = fab.absorb(fa)

        fac = Gaussian(var_names=['a', 'c'], cov=[[10, 9], [9, 10]], mean=[4, 5], log_weight=0.0)
        fbd = Gaussian(var_names=['b', 'd'], cov=[[15, 8], [8, 15]], mean=[6, 7], log_weight=0.0)

        cg = ClusterGraph([fa, fab, fac, fbd])

        # expected messages

        # from cluster 0 (fabfa) to cluster 1 (fac)
        msg_1_factor = fabfa.marginalize(vrs=['a'], keep=True)
        msg_1 = Message(msg_1_factor, 'c0#a,b', 'c1#a,c')

        # from cluster 0 (fabfa) to cluster 2 (fbd)
        msg_2_factor = fabfa.marginalize(vrs=['b'], keep=True)
        msg_2 = Message(msg_2_factor, 'c0#a,b', 'c2#b,d')

        # from cluster 1 (fac) to cluster 0 (fabfa)
        msg_3_factor = fabfa.marginalize(vrs=['a'], keep=True)
        msg_3 = Message(msg_3_factor, 'c1#a,c', 'c0#a,b')

        # from cluster 2 (fbd) to cluster 0 (fabfa)
        msg_4_factor = fabfa.marginalize(vrs=['a'], keep=True)
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

        cg.process_graph(tol=0, max_iter=1, debug=True)

        #TODO: fix this - the problem is that if more than one KLD is inf, there is no correct sorting order.
        #actual_messages = cg.messages_passed
        #for actual_message, expected_message in zip(actual_messages, expected_messages):
        #    a1_actual_message_sender_id = actual_message.sender_id
        #    a2_actual_message_received_id = actual_message.receiver_id

        #    a3_expected_message_sender_id = expected_message.sender_id
        #    a4_expected_message_received_id = expected_message.receiver_id

        #    self.assertTrue(actual_message.equals(expected_message, rtol=1e-03, atol=1e-03))
        #    self.assertEqual(actual_message, expected_message)