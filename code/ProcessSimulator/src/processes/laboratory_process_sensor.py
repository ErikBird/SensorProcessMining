import random
from typing import Union

import numpy as np
import tensorflow as tf

from ProcessSimulator.src.simulator.process_model import ProcessSimulator, ProcessStepFunction, BinetTensorSpec, \
    Preprocessing
from ProcessSimulator.src.simulator.signal_generator import perlin_noise, transition, add_noise, spaced_sawtooth, \
    add_trend, padding, add_spike_cluster

"""

Very Easy Process with Sensor Data
"""


class LaboratorySimulator(ProcessSimulator):
    DECIMALS = 2

    @property
    def __name__(self):
        return 'LaboratorySimulator'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_edge(origin='start', dest='customer_meeting')

        # Step 0 - Customer Meeting
        self.add_node(id='customer_meeting', label='Customer Meeting')
        self.add_edge(origin='customer_meeting', dest='scan')

        # Step 1 - Scan
        self.add_node(id='scan', label='Scan')
        self.add_edge(origin='scan', dest='3d_design')

        # Step 2 - 3D Design
        self.add_node(id='3d_design', label='3D Design')
        self.add_edge(origin='3d_design', dest='print_prep')

        def process_type(self) -> list:
            chosen_type = random.choice(['mold', 'cast'])
            self.process_properties['process_type'] = chosen_type
            if chosen_type == 'mold':
                return [0]
            elif chosen_type == 'cast':
                return [1]
            else:
                raise Exception('process_type not chosen correctly')

        self.add_attribute(node_id='3d_design', attribute_function=process_type,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.int8, name='product_type'))

        # Step 3 - Print Preparation

        self.add_node(id='print_prep', label='Print Prep')
        self.add_edge(origin='print_prep', dest='print')

        # Step 4 - Print

        self.add_node(id='print', label='Print')
        self.add_edge(origin='print', dest='wash')

        def sensor_temperature(self) -> list:
            result_type = np.random.choice(['cold', 'normal', 'hot'], p=[0.03, 0.92, 0.05])
            if result_type == 'cold':
                flexible_heat = random.uniform(10, 12)
                signal = perlin_noise(amplitude=1, wavelength=400, octaves=5, divisor=2, num_points=300, fixed=10,
                                      variable=flexible_heat)
            elif result_type == 'hot':
                flexible_heat = random.uniform(13, 20)
                signal = perlin_noise(amplitude=1, wavelength=400, octaves=5, divisor=2, num_points=300, fixed=30,
                                      variable=flexible_heat)
            elif result_type == 'normal':
                flexible_heat = random.uniform(10, 12)
                signal = perlin_noise(amplitude=1, wavelength=400, octaves=5, divisor=2, num_points=300, fixed=30,
                                      variable=flexible_heat)

            else:
                raise Exception('result_type %s not implemented' % result_type)

            signal = transition(signal, strength=6)

            self.process_properties['print_temperature'] = max(signal)
            return signal

        self.add_attribute(node_id='print', attribute_function=sensor_temperature,
                           attributes_signature=BinetTensorSpec(shape=(300,), dtype=tf.float32,
                                                                name='sensor_print_temp'))

        def load_curve(self) -> list:
            anomalous = np.random.choice([0, 1], p=[0.95, 0.05])
            signal = add_noise(spaced_sawtooth(30, 20, anomaly=anomalous), strength=0.05)
            self.process_properties['load_curve_anomaly'] = anomalous
            return signal

        self.add_attribute(node_id='print', attribute_function=load_curve,
                           attributes_signature=BinetTensorSpec(shape=(600,), dtype=tf.float32,
                                                                name='sensor_load_curve'))

        def skipped_layer(self) -> list:
            layer_skipped = np.random.choice([0, 1], p=[0.95, 0.05])
            self.process_properties['skipped_layer'] = layer_skipped
            return [layer_skipped]

        self.add_attribute(node_id='print', attribute_function=skipped_layer,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.int8, name='skipped_layer'))

        def print_finished(self) -> list:
            finished = np.random.choice([1, 0], p=[0.99, 0.01])
            self.process_properties['print_finished'] = finished
            return [finished]

        self.add_attribute(node_id='print', attribute_function=print_finished,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.int8, name='print_finished'))

        # Step 5 - Wash
        class CheckSensor(ProcessStepFunction):
            __destination__ = ['cure', 'fix']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['broken_sensor'] == True:
                    return self.__destination__[1]
                else:
                    return self.__destination__[0]

        def isopropanol_curve(self) -> list:
            result_type = np.random.choice(['short', 'normal', 'long'], p=[0.03, 0.94, 0.03])
            broken_sensor = np.random.choice([False, True], p=[0.95, 0.05])
            isopropanol_purity_start = np.round(np.random.uniform(low=0.85, high=1), self.DECIMALS).astype(float)
            if result_type == 'short':
                wash_duration = random.randint(50, 170)
            elif result_type == 'normal':
                wash_duration = random.randint(200, 500)
            elif result_type == 'long':
                wash_duration = random.randint(530, 620)
            else:
                raise Exception('result_type not chosen correctly')
            signal = add_trend([isopropanol_purity_start] * wash_duration, strength=0.05, negative=True)
            self.process_properties['wash_duration'] = wash_duration
            self.process_properties['broken_sensor'] = broken_sensor
            self.process_properties['isopropanol_purity_min'] = min(signal)
            if broken_sensor:
                signal = add_spike_cluster(signal)
            result = np.zeros(620)
            result[:len(signal)] = signal
            return result

        self.add_node(id='wash', label='Wash')
        self.add_decision(origin='wash', dest=CheckSensor, label='LD1: Is the sensor broken?')

        self.add_attribute(node_id='wash', attribute_function=isopropanol_curve,
                           attributes_signature=BinetTensorSpec(shape=(620,), dtype=tf.float32,
                                                                name='sensor_washing'))

        self.add_node(id='fix', label='Fix Sensor')
        self.add_edge(origin='fix', dest='cure')

        # Step 6 - Cure
        self.add_node(id='cure', label='Cure')
        self.add_edge(origin='cure', dest='material_test')

        def cure_curve(self) -> list:
            """
            120 - 300
            :param self:
            :return: Seconds
            """
            energy = np.round(np.random.uniform(low=2, high=4), self.DECIMALS).astype(float)  # mWatt
            cure_duration = random.randint(250, 300)
            signal = add_noise([energy] * cure_duration)
            self.process_properties['cure_duration'] = cure_duration
            self.process_properties['cure_energy_joule'] = sum(signal)
            result = np.zeros(300)
            result[:len(signal)] = signal
            return result

        self.add_attribute(node_id='cure', attribute_function=cure_curve,
                           attributes_signature=BinetTensorSpec(shape=(300,), dtype=tf.float32,
                                                                name='sensor_cure_energy'))

        # Step 7 - Material Test

        self.add_node(id='material_test', label='Material Test')

        class MaterialTest(ProcessStepFunction):
            __destination__ = ['cure', 'print_prep', 'end', 'cast_isolation']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['cure_energy_joule'] < 600:  # mJoule
                    return self.__destination__[0]

                flawless = model.process_properties['skipped_layer'] == False \
                           and model.process_properties['print_finished'] == True \
                           and model.process_properties['print_temperature'] < 40 \
                           and model.process_properties['print_temperature'] > 20 \
                           and model.process_properties['load_curve_anomaly'] == False \
                           and 500 > model.process_properties['wash_duration'] > 200 \
                           and model.process_properties['isopropanol_purity_min'] > 0.8

                if not flawless:
                    return self.__destination__[1]
                if model.process_properties['process_type'] == 'mold':
                    return self.__destination__[2]
                elif model.process_properties['process_type'] == 'cast':
                    return self.__destination__[3]
                else:
                    raise Exception('process_type must be mold or cast')

        self.add_decision(origin='material_test', dest=MaterialTest, label='LD2: Was the print successful?')

        # Step 8 - Cast Isolation

        self.add_node(id='cast_isolation', label='Cast Isolation')

        self.add_edge(origin='cast_isolation', dest='drying')

        # Step 9 - Drying

        self.add_node(id='drying', label='Drying')
        self.add_edge(origin='drying', dest='casting')

        def dry_temperature_curve(self) -> list:
            short = np.random.choice([True, False], p=[0.05, 0.95])
            if short:
                dry_duration = random.randint(120, 540)
            else:
                dry_duration = random.randint(600, 1320)
            self.process_properties['dry_duration'] = dry_duration
            flexible_heat = random.uniform(5, 25)
            signal = perlin_noise(amplitude=1, wavelength=10, octaves=3, divisor=2, num_points=dry_duration, fixed=30,
                                  variable=flexible_heat)
            self.process_properties['dry_temp'] = sum(signal)
            signal = transition(signal, strength=6, position='bilateral')
            signal = padding(signal, size=1320)
            self.process_properties['print_temperature'] = max(signal)

            # Compute duration for the next step already here
            # Dirty hack to have the same duration in two parallel timeseries attributes
            too_short = np.random.choice([True, False], p=[0.02, 0.98])
            if too_short:
                poly_duration = random.randint(10, 24)
            else:
                poly_duration = random.randint(25, 45)
            self.process_properties['poly_duration'] = poly_duration
            return signal

        self.add_attribute(node_id='drying', attribute_function=dry_temperature_curve,
                           attributes_signature=BinetTensorSpec(preprocessing=Preprocessing.CONVOLUTION, shape=(1320,),
                                                                dtype=tf.float32, name='sensor_dry_temp'))

        # Step 10 - Casting

        self.add_node(id='casting', label='Casting')

        self.add_edge(origin='casting', dest='polymerizing')

        # Step 11 - Polymerizing

        self.add_node(id='polymerizing', label='Polymerizing')
        self.add_edge(origin='polymerizing', dest='extract')

        def pressure_curve(self) -> list:
            lid_not_closed_properly = np.random.choice([True, False], p=[0.05, 0.95])
            poly_duration = self.process_properties['poly_duration']
            if lid_not_closed_properly:
                pressure = np.round(np.random.uniform(low=1, high=2.5), self.DECIMALS).astype(float)
            else:
                pressure = np.round(np.random.uniform(low=3, high=4), self.DECIMALS).astype(float)
            self.process_properties['pressure'] = pressure
            signal = transition(add_noise([pressure] * poly_duration), strength=1.5, position='bilateral')
            signal = padding(signal, size=45)
            return signal

        self.add_attribute(node_id='polymerizing', attribute_function=pressure_curve,
                           attributes_signature=BinetTensorSpec(shape=(45,), dtype=tf.float32, name='sensor_pressure'))

        def poly_temp(self) -> list:
            poly_duration = self.process_properties['poly_duration']
            flexible_heat = random.uniform(15, 25)
            signal = perlin_noise(amplitude=1, wavelength=10, octaves=3, divisor=2, num_points=poly_duration, fixed=40,
                                  variable=flexible_heat)
            self.process_properties['poly_temp'] = max(signal)
            signal = transition(signal, strength=2, position='bilateral')
            signal = padding(signal, size=45)
            self.process_properties['print_temperature'] = max(signal)
            return signal

        self.add_attribute(node_id='polymerizing', attribute_function=poly_temp,
                           attributes_signature=BinetTensorSpec(shape=(45,), dtype=tf.float32, name='sensor_poly_temp'))

        # Step 12 - Extract

        self.add_node(id='extract', label='Extract')

        class Extract(ProcessStepFunction):
            __destination__ = ['end', 'print']

            def destination(self, model) -> Union[int, str]:
                flawless = model.process_properties['dry_temp'] > 2400 \
                           and model.process_properties['dry_duration'] > 600 \
                           and model.process_properties['pressure'] > 3 \
                           and model.process_properties['poly_temp'] > 45 \
                           and model.process_properties['poly_duration'] > 25
                if flawless:
                    return self.__destination__[0]
                else:
                    return self.__destination__[1]

        self.add_decision(origin='extract', dest=Extract, label='LD3: Broke during extraction?')

        # Step 13 - Finish

        self.add_node(id='end', label='End')

        class Finish(ProcessStepFunction):
            __destination__ = ['start']

            def destination(self, model) -> Union[int, str]:
                model.reset()
                return model.last_visited_step_id

        self.add_decision(origin='end', dest=Finish)
