from PIL import Image as PIL_Image
import re


def make_sepset_node_name(node_a_name, node_b_name):
    return 'sepset__' + '__'.join(sorted([node_a_name, node_b_name]))


def change_cluster_graph_edge_color(graph, node_a_name, node_b_name, new_color):
    """
    Change the color of an edge in a cluster graph graphviz object. Note that the cluster graph has cluster and sepset
    nodes. The 'edge' of which the color is changed in this function is actually two edges in the graphviz object (the
    edge between node_a and the sepset node and the edge between the sepset node and node_b).

    :param graph: The graphviz graph
    :param node_a_name:
    :param node_b_name:
    :param new_color:
    :return:

    """

    sepset_node_name = make_sepset_node_name(node_a_name, node_b_name)
    change_graph_edge_color(graph=graph, node_a_name=node_a_name, node_b_name=sepset_node_name, new_color=new_color)
    change_graph_edge_color(graph=graph, node_a_name=node_b_name, node_b_name=sepset_node_name, new_color=new_color)


def check_non_overlapping_substring_presence(substring_a, substring_b, string):
    """
    Check that both sub-strings can be found in separate locations in strings

    :param substring_a: The one substring to check for.
    :param substring_b: The other substring to check for.
    :param string: The string to check in.
    :return: Result of check for independent string presence.

     Examples:
        substring_a = abc
        substring_b = 123
        string = abc123 -- def
        returns False

        substring_a = abc
        substring_b = 123
        string = abc -- 123
        returns True
    """

    if substring_a in substring_b:
        if substring_a not in string:
            return False
        string = string.replace(substring_b, '')
        if substring_a in string:
            return True
    if substring_b in substring_a:
        if substring_b not in string:
            return False
        string = string.replace(substring_a, '')
        if substring_b in string:
            return True
    return (substring_a in string) and (substring_b in string)


def change_graph_edge_color(graph, node_a_name, node_b_name, new_color):
    """
    Change the edge color of the edge between two nodes in a graphviz object.

    :param graph: The graphviz graph
    :param node_a_name: The one node
    :param node_b_name: The other node
    :param new_color: The new color of the edge (i.e 'green', 'blue', 'red')
    :return:
    """

    edge_found = False
    for i, s in enumerate(graph.body):
        if '--' in s:
            if check_non_overlapping_substring_presence(node_a_name, node_b_name, s):
                pattern = f'(\\t?"?{node_a_name}"?\s--\s"?{node_b_name}"?\s\[.*color=)([\S]*\s)(.*)'
                new_s = re.sub(pattern, f'\g<1>{new_color} \g<3>', s)
                # TODO: improve this (it's a bit hacky)
                if new_s == s:
                    pattern = f'(\\t?"?{node_b_name}"?\s--\s"?{node_a_name}"?\s\[.*color=)([\S]*\s)(.*)'
                    new_s = re.sub(pattern, f'\g<1>{new_color} \g<3>', s)

                graph.body[i] = new_s
                edge_found = True
    if not edge_found:
        raise ValueError(
            f'cannot change colour of not existing edge: no edge between node {node_a_name} and node {node_b_name} in graph')


def change_graph_node_color(graph, node_name, new_color):
    """
    Change the fill color of a node in a cluster graph graphviz object.

    :param graph:  The graphviz graph
    :param node_name: The name of the node which colour should be changed.
    :param new_color: The new fill color of the node (i.e 'green', 'blue', 'red')
    :return:
    """
    node_found = False
    for i, s in enumerate(graph.body):
        if '--' not in s:  # not edge
            if node_name in s:
                graph.body[i] = re.sub(f'(\\t?"?{node_name}"?\s\[.*fillcolor=)([\S]*\s)(.*)',f'\g<1>{new_color} \g<3>', s)
                node_found = True
    if not node_found:
        raise ValueError(f'cannot change colour of not existing node: no node {node_name} in graph')


def graph_to_pil_image(graph):
    graph.render('/tmp/test.gv', view=False)
    pil_image = PIL_Image.open('/tmp/test.gv.png')
    pil_image_copy = pil_image.copy()
    pil_image.close()
    return pil_image_copy


def add_message_pass_animation_frames(graph, frames, node_a_name, node_b_name):
    new_color = 'red'
    change_graph_node_color(graph=graph,
                            node_name=node_a_name,
                            new_color=new_color)
    frames.append(graph_to_pil_image(graph))

    change_cluster_graph_edge_color(graph=graph,
                                    node_a_name=node_a_name,
                                    node_b_name=node_b_name,
                                    new_color=new_color)
    frames.append(graph_to_pil_image(graph))

    change_graph_node_color(graph=graph,
                            node_name=node_b_name,
                            new_color=new_color)
    frames.append(graph_to_pil_image(graph))
    change_graph_node_color(graph=graph,
                            node_name=node_a_name,
                            new_color='white')
    frames.append(graph_to_pil_image(graph))

    change_cluster_graph_edge_color(graph=graph,
                                    node_a_name=node_a_name,
                                    node_b_name=node_b_name,
                                    new_color='black')
    frames.append(graph_to_pil_image(graph))
    change_graph_node_color(graph=graph,
                            node_name=node_b_name,
                            new_color='white')
    frames.append(graph_to_pil_image(graph))
