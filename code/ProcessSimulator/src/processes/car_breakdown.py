import random
from typing import Union

import numpy as np
import tensorflow as tf

from ProcessSimulator.src.simulator.process_model import ProcessSimulator, ProcessStepFunction, \
    BinetTensorSpec, \
    Preprocessing, SensorAnomalies
from ProcessSimulator.src.simulator.signal_generator import perlin_noise

"""

Very Easy Process with Sensor Data
"""


class CarBreakdownSimulator(ProcessSimulator):
    DECIMALS = 2
    def __repr__(self):
        return 'CarBreakdownSimulator'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        class CheckControlLight(ProcessStepFunction):
            __destination__ = ['drive', 'diagnose']

            def destination(self, model) -> Union[int, str]:

                if model.process_properties['engine_control_light'] == 1:
                    model.process_properties['issue'] = np.random.choice(['battery', 'thermostat', 'air_flow'], p=[0.4, 0.3, 0.3])
                    return self.__destination__[1]
                else:
                    return self.__destination__[0]

        self.add_edge(origin='start', dest='check_control_light')
        self.add_node(id='check_control_light', label='engine control light on?')
        self.add_edge(origin='check_control_light', dest=CheckControlLight)

        def engine_control_light(self) -> list:
            light_on = np.random.choice([1, 0], p=[0.70, 0.30])
            self.process_properties['engine_control_light'] = light_on
            return [light_on]

        self.add_attribute(node_id='check_control_light', attribute_function=engine_control_light,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.int8, name='control_light'))

        self.add_node(id='drive', label='Drive')
        self.add_edge(origin='drive', dest='end')

        self.add_node(id='diagnose', label='Diagnose')

        def battery_energy(self) -> list:
            if self.process_properties['issue'] == 'battery':
                battery_energy = 0
            else:
                battery_energy = 2
            signal = perlin_noise(amplitude=1, wavelength=2000, octaves=5, divisor=2, num_points=100, fixed=battery_energy,
                                  variable=5)
            return signal

        self.add_attribute(node_id='diagnose', attribute_function=battery_energy,
                           attributes_signature=BinetTensorSpec(shape=(100,), dtype=tf.float32, name='sensor_battery_energy',
                                                                sensor_anomalies=[SensorAnomalies.NOISE, SensorAnomalies.SPIKE]))

        def air_flow_meter(self) -> list:
            signal = perlin_noise(amplitude=1, wavelength=2000, octaves=5, divisor=2, num_points=200, fixed=0,
                                  variable=10)
            if self.process_properties['issue'] == 'air_flow':
                signal = add_stuck_at_zero(signal)
            return signal

        self.add_attribute(node_id='diagnose', attribute_function=air_flow_meter,
                           attributes_signature=BinetTensorSpec(shape=(200,), dtype=tf.float32, name='sensor_air_flow_meter',
                                                                sensor_anomalies=[SensorAnomalies.NOISE, SensorAnomalies.SPIKE]))
        def thermostat(self) -> list:
            if self.process_properties['issue'] == 'thermostat':
                return [1]
            else:
                return [0]

        self.add_attribute(node_id='diagnose', attribute_function=thermostat,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.int8, name='thermostat'))

        class IssueFound(ProcessStepFunction):
            __destination__ = ['charge_battery', 'change_thermostat', 'change_flow_meter']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['issue'] == 'battery':
                    return self.__destination__[0]
                elif model.process_properties['issue'] == 'thermostat':
                    return self.__destination__[1]
                elif model.process_properties['issue'] == 'air_flow':
                    return self.__destination__[2]
                else:
                    raise NotImplementedError

        self.add_edge(origin='diagnose', dest='found_issue')
        self.add_node(id='found_issue', label='Found Issue?')
        self.add_edge(origin='found_issue', dest=IssueFound)
        self.add_node(id='charge_battery', label='Charge Battery')
        self.add_edge(origin='charge_battery', dest='listen_engine')
        self.add_node(id='change_thermostat', label='Change Thermostat')
        self.add_edge(origin='change_thermostat', dest='listen_engine')
        self.add_node(id='change_flow_meter', label='Change Air Flow Meter')
        self.add_edge(origin='change_flow_meter', dest='listen_engine')

        self.add_node(id='listen_engine', label='Listen to Engine')
        self.add_edge(origin='listen_engine', dest='loud')
        self.add_node(id='loud', label='Too Loud?')

        def decibel(self) -> list:
            loud = np.random.choice([True, False], p=[0.70, 0.30])
            self.process_properties['loud'] = loud
            if loud:
                base_level = 70
            else:
                base_level = 60
            signal = perlin_noise(amplitude=1, wavelength=2000, octaves=5, divisor=2, num_points=100, fixed=base_level,
                                  variable=30)
            return signal

        self.add_attribute(node_id='listen_engine', attribute_function=decibel,
                           attributes_signature=BinetTensorSpec(shape=(100,), dtype=tf.float32, name='sensor_decibel',
                                                                sensor_anomalies=[SensorAnomalies.NOISE, SensorAnomalies.SPIKE]))

        class TooLoud(ProcessStepFunction):
            __destination__ = ['repair_shop', 'drive']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['loud']:
                    return self.__destination__[0]
                else:
                    return self.__destination__[1]

        self.add_edge(origin='loud', dest=TooLoud)

        self.add_node(id='repair_shop', label='Call Repair Shop')
        self.add_edge(origin='repair_shop', dest='end')
        self.add_node(id='end', label='End')

        class Finish(ProcessStepFunction):
            __destination__ = ['start']

            def destination(self, model) -> Union[int, str]:
                model.reset()
                return model.last_visited_step_id

        self.add_edge(origin='end', dest=Finish)
