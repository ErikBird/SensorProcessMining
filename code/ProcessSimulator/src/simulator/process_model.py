import subprocess
import uuid
from collections import Callable, defaultdict, deque, Counter
from typing import Union, Dict, List
from enum import Enum
import IPython
import ipycytoscape
import networkx as nx
import pydot
import tensorflow as tf
import abc
import random
import numpy as np
import dot2tex
import matplotlib.pyplot as plt
import os
import imageio
from tqdm import tqdm

from ProcessSimulator.src.simulator.signal_generator import add_noise, add_spike
from utils import get_graphviz_cache_path, GifReader, get_animation_output_path


class SensorAnomalies(Enum):
    NOISE = add_noise
    SPIKE = add_spike


class Preprocessing(Enum):
    """
    Optional preprocessing information to be send to BiNet in each attribute spec
    """
    ID = 'id'
    CONVOLUTION = 'convolution'
    DEEP_CNN = 'ml_cnn'
    RNN = 'rnn'
    KALMAN = 'kalman'
    NORMALIZE = 'normalize'
    HAMPEL = 'hampel'
    DISABLE = 'disable'
    DEFAULT = 'default'
    DEEP_CNN_NO_POOL = 'ml_cnn_no_pool'
    DEEP_FFN = 'ml_ffn'
    NORMALIZED_RNN = 'norm_rnn'
    NORMALIZED_CNN = 'norm_cnn'
    EXTENDED_KALMAN = 'extended_kalman'


class DatasetStage(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'
    TEST_DECISION = 'test_only_decisions'


class BinetTensorSpec(tf.TensorSpec):
    """
    Extension of the Tensorflow TensorSpec Class to be able to add custom attributes

    The class is used to describe the shape of the data and how the data can be further processed
    """
    preprocessing: Preprocessing
    sensor_anomalies: Union[SensorAnomalies, List[SensorAnomalies]]

    def __init__(self, preprocessing=None, sensor_anomalies=None, **kw):
        super().__init__(**kw)
        self.preprocessing = preprocessing
        self.sensor_anomalies = sensor_anomalies

    def set_preprocessing(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing


class ProcessStepFunction:
    """
    This class represents a node decision in the process graph

    All possible destinations should be defined in the __destination__ attribute.
    This is necessary to visualize all graph edges without computing them.

    The decision logic should be implemented in the destination function.
    The logic can be arbitrary complex.
    The process_model argument provides an opportunity to access the process_properties of the model.

    Example:
    ```
    class NodeDecision(ProcessStepFunction):
        __destination__ = ['node_one', 'node_two']

        def destination(self, model) -> Union[int, str]:
            return random.choice(self.__destination__)
    ```
    """
    # Important to update the metaclass to be able to determine the class type in the generator
    __metaclass__ = abc.ABCMeta
    # List of all destinations, the ProcessStep can refer to
    __destination__ = []
    # Random might be used by attributes or decisions
    random.seed(11)

    def destinations(self) -> list:
        """
        :return: List of all destination ids the Node Decision can refer to.
        """
        return self.__destination__

    @abc.abstractmethod
    def destination(self, process_model) -> Union[int, str]:
        """
        The method to compute a Node Decision
        :param process_model: The process model context, the decision takes place in.
        :return: The Destination ID of the decision outcome
        """
        raise NotImplementedError


class AttributeDataExhaustException(Exception):
    """Exception to signal the generator to terminate due to an exhaustion in available data"""
    pass


class ProcessSimulator:
    """
    This class can represent an arbitrary process wich follows the logic of a directed graph.

    The main goal is to provide an interface to a datastream of a simulation of the process.

    Each Node in this graph can have arbitrary numeric attributes which can also be computed during the simulation.

    The Node destination can also be dependent on a just-in-time computed function.
    """

    def __init__(self,
                 iteration_limit: int = 100000,
                 input_window_width: int = 5):
        """
        :param edges: Dict with key: id, value: destination_id of Nodes in this Graph
        :param nodes: Dict with key: id, value: node_label to define Graph Nodes
        :param iteration_limit: The maximum number of simulated steps the generator should yield before terminating
        :param input_window_width: the window size of the input which should be accessible for the prediction
        """
        # Defininion of Directed Process Graph
        self.edges = {}  # key: id, value: destination_id
        self.nodes = {'start': 'Start'}  # key: id, value: node_label
        self.decision_labels = {}
        # Define Attributes of Process Steps
        self.attributes = defaultdict(
            dict)  # Dict of all attributes that are set in this step. Attributes are usually callables
        # output signature of the generator for Tensorflow
        self.output_signature: list = []
        # Track the current position in the process graph
        self.last_visited_step_id = 'start'
        # Storage for information within one process iteration ( cleaned after each iteration)
        self.process_properties = {}
        # Storage for information persistent during the whole generation precess
        self.persistent_properties = {}
        # Number of iterations to be generated before terminating
        self.iteration_limit = iteration_limit
        # size of accessible history
        self.window_width = input_window_width
        # List of the last process steps and the next step which should be predicted
        # This data structure automatically deletes steps which are not inside of the input window anymore
        self.history = deque(maxlen=self.window_width + 1)
        self.dataset_stage = DatasetStage.TRAIN

    @property
    def node_depth(self) -> int:
        """
        :return: The number of Nodes in the Graph
        """
        return len(set(self.edges.keys()))

    @property
    def get_decisions(self) -> list:
        one_hot_func = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.node_names,
                                                                               output_mode='one_hot')

        def check_decision(item):
            return isinstance(item, type) and issubclass(item, ProcessStepFunction)

        # end node is an exception bc it is not really a decision.
        # It is only a decision in the system bc. it needs to clean the instance information during its execution
        return [one_hot_func([node]) for node in self.nodes.keys() if
                check_decision(self.edges[node]) and node != 'end']

    @property
    def node_names(self):
        """
        :return: All defined Node IDs in the Graph
        """
        return tf.constant(list(self.nodes.keys()))

    @property
    def attribute_names(self):
        """
        :return: All defined Attribute Names of all Nodes in the Graph
        """
        return [spec.name for spec in self.output_signature]

    @abc.abstractmethod
    def __name__(self):
        raise NotImplementedError

    def __str__(self):
        """
        :return: The DOT Graph representation
        """
        return self.get_dot_graph().to_string()

    def get_signature_by_name(self, name):
        """

        :param name:
        :return:
        """
        for signature in self.output_signature:
            if signature.name == name:
                return signature
        return None

    def add_edge(self, origin: Union[str, int],
                 dest: Union[str, int]):
        """

        :param origin:
        :param dest:
        :return:
        """
        assert not origin == dest, 'No loops allowed!'
        self.edges[str(origin).lower()] = str(dest).lower()

    def add_decision(self, origin: Union[str, int],
                     dest: ProcessStepFunction, label=''):
        """

        :param origin:
        :param dest:
        :return:
        """
        assert not origin == dest, 'No loops allowed!'

        self.edges[str(origin).lower()] = dest
        self.decision_labels[str(origin).lower()] = label

    def add_node(self, id: Union[str, int], label: str):
        """

        :param id:
        :param label:
        :return:
        """
        assert str(id) not in self.nodes.keys(), 'Node with given id already exists!'
        self.nodes[str(id)] = label

    def make_spec_name_unique(self, spec: BinetTensorSpec) -> BinetTensorSpec:
        """
        Needs to generate a new BinetTensorSpec since TensorSpec is not mutable
        :param spec:
        :return:
        """

        def enumerate_name(name: str, step: int):
            if name.lower() + '_' + str(step) in self.attribute_names:
                return enumerate_name(name=name.lower(), step=step + 1)
            else:
                return name.lower() + '_' + str(step)

            # shape=(self.window_width) + spec.shape

        return BinetTensorSpec(shape=tuple([None] + spec.shape), dtype=spec.dtype,
                               name=enumerate_name(name=spec.name, step=0),
                               preprocessing=spec.preprocessing, sensor_anomalies=spec.sensor_anomalies)

    def add_attribute(self,
                      node_id: Union[Union[str, int], List[Union[str, int]]],
                      attribute_function: Callable[..., list],
                      attributes_signature: BinetTensorSpec):
        """

        :param node_id:
        :param attribute_function:
        :param attributes_signature:
        :return:
        """
        assert attributes_signature.name is not None, "the attributes_signature spec must contain the attribute 'name'"

        unique_signature = self.make_spec_name_unique(attributes_signature)
        name = unique_signature.name

        self.output_signature.append(unique_signature)
        if isinstance(node_id, list):
            for id in node_id:
                self.attributes[str(id).lower()].update({name: attribute_function})
        else:
            self.attributes[str(node_id).lower()].update({name: attribute_function})

    def generate_next_step(self):
        def execute_callable(item):
            if isinstance(item, type) and issubclass(item, ProcessStepFunction):
                return item.destination(item, self)
            elif callable(item):
                return item(self)
            else:
                return item

        current_step_id = execute_callable(self.edges[self.last_visited_step_id])
        current_step_id = str(current_step_id).lower()
        attribute_values = []
        for name in self.attribute_names:
            if name == 'id':
                continue
            if name in self.attributes[current_step_id].keys():
                sensor_anomalies = self.get_signature_by_name(name).sensor_anomalies
                if not sensor_anomalies:
                    attr_value = execute_callable(self.attributes[current_step_id][name])
                    attribute_values.append(attr_value)
                elif isinstance(sensor_anomalies, list):
                    data = execute_callable(self.attributes[current_step_id][name])
                    for anomaly in sensor_anomalies:
                        data = anomaly(data)
                    attribute_values.append(data)
                else:
                    data = execute_callable(self.attributes[current_step_id][name])
                    data = sensor_anomalies(data)
                    attribute_values.append(data)
            else:
                attribute_values.append(tf.zeros(self.get_signature_by_name(name).shape[1:]))

        self.last_visited_step_id = current_step_id
        output = [current_step_id] + attribute_values
        return output

    def generator(self):
        """

        :return:
        """
        print('Generate %s process iterations' % self.iteration_limit)

        for _ in tqdm(range(self.iteration_limit)):
            try:
                yield from self.get_next()
            except AttributeDataExhaustException:
                print('stop generation due to exhausted attribute data')
                break

    def sequence_generator(self, process_instances: int = 1000, dataset_stage: DatasetStage = DatasetStage.TRAIN,
                           maximal_recursion_depth: int = 30):
        one_hot_func = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.node_names,
                                                                               output_mode='one_hot')

        self.dataset_stage = dataset_stage
        print('\n_____Generate Dataset_____')
        for _ in tqdm(range(process_instances)):
            try:
                steps = []
                for _ in range(maximal_recursion_depth):
                    step = self.generate_next_step()
                    steps.append(step)
                    if step[0] == 'end':
                        break
                # Reset to start from the beginning
                # Explicitly needed to be called bc. 'end' might not be reached
                self.reset()
                input_labels = []
                for input in steps:
                    seq = [one_hot_func([input[0]])] + input[1:]
                    input_labels.append(seq)
                    # input_labels.append([[input[0]]])  # + input[1:])
                yield input_labels
            except AttributeDataExhaustException:
                print('stop generation due to exhausted attribute data')
                break

    def generator_anomalies(self):
        print('Generate %s process iterations' % self.iteration_limit)
        for _ in tqdm(range(self.iteration_limit)):
            try:
                yield from self.get_next_anomalies()
            except AttributeDataExhaustException:
                print('stop generation due to exhausted attribute data')
                break

    def get_start_output(self):
        attribute_values = []
        for name in self.attribute_names:
            if name == 'id':
                continue
            attribute_values.append(tf.zeros(self.get_signature_by_name(name).shape[1:]))
        empty_output = ['start'] + attribute_values
        return empty_output

    def get_next_anomalies(self):
        """

        :return:
        """
        one_hot_func = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.node_names,
                                                                               output_mode='one_hot')

        def yield_based_on_current_history(anomaly):
            input_labels = []
            for input in list(self.history)[:-1]:
                input_labels.append([one_hot_func([input[0]])] + input[1:])

            shaped_index = []
            for index in range(len(input_labels[0])):
                attribute = []
                for event in input_labels:
                    attribute.append(event[index])
                shaped_index.append(tf.stack(attribute))

            out = tuple([tuple(shaped_index), (one_hot_func([list(self.history)[-1][0]]), anomaly)])
            yield out

        # Fill the history with empty output on process start
        if self.last_visited_step_id == 'start':
            empty_output = self.get_start_output()
            for _ in range(self.window_width):
                self.history.append(empty_output)
        ## Anomalies
        # Skip
        skip_anomaly = np.random.choice([True, False], p=[0.5, 0.5])
        if skip_anomaly:
            missed_step = self.generate_next_step()[1]
            self.history.append(self.generate_next_step()[1])
            yield from yield_based_on_current_history(skip_anomaly)
            last_step = self.history[-1]
            self.history[-1] = missed_step
            self.history.append(last_step)
        '''
        # Replace
        if np.random.choice([False, True], p=[0.9, 0.1]):
            random_index = random.randint(0, self.window_width - 1)
            random_index2 = random.randint(0, self.window_width - 1)
            swap_cache = self.history[random_index]
            self.history[random_index] = self.history[random_index2]
            self.history[random_index2] = swap_cache
        '''
        self.history.append(self.generate_next_step()[1])
        yield from yield_based_on_current_history(anomaly=False)

    def get_next(self):
        """

        :return:
        """
        # Fill the history with empty output on process start
        if self.last_visited_step_id == 'start':
            empty_output = self.get_start_output()
            for _ in range(self.window_width):
                self.history.append(empty_output)
        self.history.append(self.generate_next_step()[1])
        one_hot_func = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.node_names,
                                                                               output_mode='one_hot')
        input_labels = []
        for input in list(self.history)[:-1]:
            input_labels.append([one_hot_func([input[0]])] + input[1:])

        shaped_index = []
        for index in range(len(input_labels[0])):
            attribute = []
            for event in input_labels:
                attribute.append(event[index])
            shaped_index.append(tf.stack(attribute))

        out = tuple([tuple(shaped_index), one_hot_func([list(self.history)[-1][0]])])
        yield out

    def reset(self):
        """

        :return:
        """
        self.process_properties = {}
        self.last_visited_step_id = 'start'

    def get_output_signature(self):
        """

        :return:
        """

        signature = (
            tuple([BinetTensorSpec(preprocessing=Preprocessing.ID, shape=(None, self.node_depth + 1),
                                   dtype=tf.int32, name='id')] + self.output_signature),
            BinetTensorSpec(shape=(self.node_depth + 1), dtype=tf.int32, name='label'))
        '''signature = [BinetTensorSpec(preprocessing=Preprocessing.ID, shape=(self.node_depth + 1),
                                   dtype=tf.int32, name='id')] + self.output_signature'''
        return signature

    def get_signature_with_timesteps(self, timesteps: int):
        """

        :return:
        """

        def get_spec_with_timesteps(spec: BinetTensorSpec, timesteps: int) -> BinetTensorSpec:
            return BinetTensorSpec(shape=tuple([timesteps] + spec.shape[1:]), dtype=spec.dtype,
                                   name=spec.name,
                                   preprocessing=spec.preprocessing, sensor_anomalies=spec.sensor_anomalies)

        signature = (
            tuple([BinetTensorSpec(preprocessing=Preprocessing.ID, shape=(timesteps, self.node_depth + 1),
                                   dtype=tf.int32, name='id')] +
                  [get_spec_with_timesteps(spec=spec, timesteps=timesteps) for spec in self.output_signature]),
            BinetTensorSpec(shape=(self.node_depth + 1), dtype=tf.int32, name='label'))
        '''signature = [BinetTensorSpec(preprocessing=Preprocessing.ID, shape=(self.node_depth + 1),
                                   dtype=tf.int32, name='id')] + self.output_signature'''
        return signature

    def get_dot_graph(self, probabilities=False, conditional_probability=False, highlight=[], sensor_values={}):
        """

        :return:
        """
        # Store properties to reset them after the visualization
        # They can change since sensor attributes are computed to visualize example data
        tmp_process_properties = self.process_properties
        tmp_persistent_properties = self.persistent_properties
        if probabilities:
            probabilities = self.compute_graph_probabilities()
            print(probabilities)
            if conditional_probability:
                unique_origins = set([orig for orig, dest in probabilities.keys()])
                total_per_origin = {}
                for origin in unique_origins:
                    total_per_origin[origin] = sum([v for k, v in probabilities.items() if k[0] == origin])
                probabilities = {k: v / total_per_origin[k[0]] for k, v in probabilities.items()}

        def draw_edge(origin, destination, label='', probability_key=None):
            if isinstance(destination, type) and issubclass(destination, ProcessStepFunction):
                if not origin == 'end' and not destination == 'start':
                    draw_edge(str(origin), str(destination))
                    # highlight
                    graph.add_node(pydot.Node(str(destination),
                                              xlabel=self.decision_labels[str(origin).lower()],
                                              forcelabels=True,
                                              label='X',
                                              fontsize=18,
                                              shape='diamond'))
                for dest in list(destination.destinations(destination)):
                    assert isinstance(dest, int) or isinstance(dest, str), 'destination must be int or str'
                    draw_edge(str(destination), str(dest), probability_key=(str(origin), str(dest)))
            else:
                if not origin == 'end' and not destination == 'start':
                    if probabilities:
                        try:
                            if probability_key:
                                label = round(probabilities[probability_key], 2)

                            else:
                                label = round(probabilities[(origin, destination)], 2)
                        except:
                            label = ''
                    graph.add_edge(
                        pydot.Edge(str(origin), str(destination), color='black',
                                   arrowhead='normal', label=label))

        graph = pydot.Dot()
        # Add nodes
        for id, label in self.nodes.items():
            if str(id) in highlight:
                graph.add_node(pydot.Node(str(id), label=label, shape='box', fillcolor='bisque', style='filled'))
            else:
                graph.add_node(pydot.Node(str(id), label=label, shape='box'))

        # Add edges
        for origin, destination in self.edges.items():
            draw_edge(origin, destination)

        # Add Attributes
        for node, attr in self.attributes.items():
            for label, function in attr.items():
                graph.add_edge(pydot.Edge(str(node), label, color='black', arrowhead='box', style='dotted'))
                if 'sensor' in label:
                    fig = plt.figure(figsize=(3, 1))
                    if label in sensor_values.keys():
                        plt.plot(sensor_values[label])
                    else:
                        plt.plot(function(self))
                    get_graphviz_cache_path().mkdir(parents=True, exist_ok=True)
                    plt.savefig(get_graphviz_cache_path() / str(label + '.svg'), format='svg')
                    plt.close(fig)  # Don't show the plot
                    graph.add_node(pydot.Node(label, label=label, shape='note', style='filled', fillcolor="lightgrey",
                                              image=str(get_graphviz_cache_path() / str(label + '.svg')), labelloc='b',
                                              width=2.3, height=1.3))
                else:
                    graph.add_node(pydot.Node(label, label=label, shape='note', style='filled', fillcolor="lightgrey"))

        graph.add_node(pydot.Node('start', label=self.nodes['start'], shape='circle', penwidth=2))
        graph.add_node(pydot.Node('end', label=self.nodes['end'], shape='doublecircle', penwidth=2))
        self.process_properties = tmp_process_properties
        self.persistent_properties = tmp_persistent_properties
        return graph

    def visualize_sequence(self, sequence: list = [], flow_description=[], sensor_description=[]):
        sensor_index = 0
        i_layer = tf.keras.layers.StringLookup(vocabulary=self.node_names, invert=True)
        for case_index, (label, case) in tqdm(
                enumerate(zip(flow_description, sequence))):
            ims = []
            sensor_values = {}
            name = self.__name__ + '_' + str(case_index) + '_' + label
            if len(sensor_description) > 0:
                name += '_SensorAnom'
            for step_index, step in enumerate(case):
                id = str(i_layer([np.argmax(step[0])]).numpy()[0].decode('UTF-8'))
                sensor_desc = sensor_description[sensor_index]
                sensor_index += 1
                for index, spec in enumerate(self.get_signature_with_timesteps(timesteps=1)[0]):
                    if 'sensor' in spec.name:
                        if tf.is_tensor(step[index]):
                            sensor_value = step[index].numpy()
                        else:
                            sensor_value = np.array(step[index])
                        if spec.name not in sensor_values.keys() or sensor_value.nonzero()[0].size != 0:
                            sensor_values[spec.name] = sensor_value

                graph = self.get_dot_graph(highlight=[id], sensor_values=sensor_values)
                image = graph.create_png()
                ims.append(imageio.imread(image))
                single_image_folder = get_animation_output_path() / name
                single_image_folder.mkdir(parents=True, exist_ok=True)
                with open(single_image_folder / str(str(step_index + 1)+'-'+ '-'.join(sensor_desc) + '.png'), 'wb') as f:
                    f.write(image)

            imageio.mimwrite(get_animation_output_path() / str(name + ".gif"), ims, fps=1)
            # subprocess.run(['open', "animation.gif"], check=True)

    def compute_graph_probabilities(self) -> Dict:
        """
        Compute
        :return:
        """
        outcome = []  # List of Edges (from, to)

        iterations = 100000
        for _ in tqdm(range(iterations)):
            outcome.append((self.last_visited_step_id, self.generate_next_step()[0]))
        # Counter counts the number of occurrences in a list - output: dict
        counts = Counter(outcome)
        probabilities = {k: v / iterations for k, v in counts.items()}
        return probabilities

    def save_graph_png(self, path='output', probabilities=False, conditional_probability=False):
        """

        :param conditional_probability:
        :param probabilities:
        :param path:
        :return:
        """
        graph = self.get_dot_graph(probabilities=probabilities, conditional_probability=conditional_probability)
        image = graph.create_png()
        if not os.path.exists(path):
            os.makedirs(path)
        with open(str(path) + '.png', 'wb') as f:
            f.write(image)

    def save_graph_svg(self, path='output', probabilities=False, conditional_probability=False):
        """

        :param conditional_probability:
        :param probabilities:
        :param path:
        :return:
        """
        graph = self.get_dot_graph(probabilities=probabilities, conditional_probability=conditional_probability)
        image = graph.create_svg()
        if not os.path.exists(path):
            os.makedirs(path)
        with open(str(path) + '.svg', 'wb') as f:
            f.write(image)
        # graph.write_png(path)

    def save_graph_tikz(self, path='output', probabilities=False, conditional_probability=False):
        graph = self.get_dot_graph(probabilities=probabilities, conditional_probability=conditional_probability)
        texcode = dot2tex.dot2tex(graph.to_string(), format='pgf', crop=True, figonly=True)
        with open(path + '.pgf', 'w') as f:
            f.write(texcode)

    def print_graph(self, probabilities=False, conditional_probability=False):
        """

        :return:
        """
        graph = self.get_dot_graph(probabilities=probabilities, conditional_probability=conditional_probability)
        image = graph.create_png()
        ipython_image = IPython.display.Image(data=image, format='png')
        IPython.display.display(ipython_image)

    def print_graph_interactive(self):
        """

        :return:
        """
        graph = self.get_dot_graph()
        nx_graph = nx.drawing.nx_pydot.from_pydot(graph)
        cytoscapeobj = ipycytoscape.CytoscapeWidget()
        cytoscapeobj.graph.add_graph_from_networkx(nx_graph)
        return cytoscapeobj
