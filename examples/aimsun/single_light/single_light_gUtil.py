import numpy as np
from gym.spaces import Box, Tuple, Discrete
import math

from flow.envs import Env
from flow.networks import Network

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3344],
                         # 'observed_nodes': [3386, 3371, 3362, 3373],
                         'num_incoming_edges_per_node': 4,
                         'num_detector_types': 4,
                         'num_measures': 2,
                         'detection_interval': (0, 15, 0),
                         'statistical_interval': (0, 15, 0),
                         'replication_list': ['Replication 8050297',  # 5-11
                                              'Replication 8050315',  # 10-14
                                              'Replication 8050322'
                                              ]}  # 14-21
# the replication list should be copied in load.py

RLLIB_N_ROLLOUTS = 3  # copy from train_rllib.py

np.random.seed(1234567890)


def set_cycle_length(rep_name, simulation_time):

    rep_name = str(rep_name)
    rep_nums = [8050297, 8050315, 8050322]
    cycle_dict = {'8050297': [110, 90, 120, 90], '8050315': [90, 105], '8050322': [105, 120, 105, 90]}
    time_dict = {'8050297': [2700, 5400, 14400], '8050315': [5400], '8050322': [7200, 18000, 23400]}
    for i in range(len(rep_nums)):
        if rep_name == str(rep_nums[i]):
            cycle_list = cycle_dict[(rep_name)]
            time_list = time_dict[(rep_name)]
            for i in range(len(time_list)):
                if simulation_time < time_list[i]:
                    cycle = (cycle_list[i])
                    break
                elif simulation_time >= time_list[-1]:
                    cycle = (cycle_list[-1])
                    break

    return cycle


def rescale_bar(array, NewMin, NewMax):
    rescaled_action = []
    OldMin = 0
    OldMax = 70
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    for OldValue in array:
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        rescaled_action.append(NewValue)
    return rescaled_action


def rescale_act(actions_array, target_value, current_value):
    rescaled_actions = []
    target_value = round(target_value)
    for duration in actions_array:
        if current_value == 0:
            new_action = 0
        else:
            new_action = math.ceil(target_value*duration/current_value)
        rescaled_actions.append(int(new_action))
    if sum(rescaled_actions) > target_value:
        x = sum(rescaled_actions) - target_value
        rescaled_actions[-1] = int(rescaled_actions[-1] - x)
    return rescaled_actions


class SingleLightEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for param in ADDITIONAL_ENV_PARAMS:
            if param not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(param))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.episode_counter = 0
        self.detection_interval = self.additional_params['detection_interval'][1]*60  # assuming minutes for now
        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])
        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])
        self.k.traffic_light.set_replication_seed(np.random.randint(2e9))

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.cycle = 0
        self.max_duration = 60
        self.minimum_green = 3
        self.sum_interphase = 18

        # reset_phase_durations
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.change_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        self.past_cumul_queue = {}
        self.detector_lane = {}
        self.observed_phases = []
        self.phases = []
        self.sum_barrier = []
        self.current_phase_timings = []

        # hardcode maxout values maxd_dict = {'control_id':'phase_maxout'}
        # Suggestion move this to control plan
        e = p1 = p4 = [23, 42, 23, 42, 23, 42, 23, 42]
        p2 = p23 = p3 = [28, 62, 28, 62, 28, 62, 28, 62]
        self.maxd_dict = {8050297: [e, p4, p2, p4],
                          8050315: [p4, p1],
                          8050322: [p1, p3, p23, p1, p4]}

        # get cumulative queue lengths
        for node_id in self.target_nodes:
            self.node_id = node_id
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            self.edge_detector_dict[node_id] = {}
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                through = detector_dict['through']
                right = detector_dict['right']
                left = detector_dict['left']
                advanced = detector_dict['advanced']
                type_map = {"through": through, "right": right, "left": left, "advanced": advanced}

                detector_lane = self.k.traffic_light.get_detector_lanes(edge_id)
                for _, (d_id, lane) in enumerate(detector_lane.items()):
                    self.detector_lane[d_id] = lane
                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_cumul_queue[edge_id] = 0

            # get control_id and # of rings
            # print(self.edge_detector_dict)
            self.control_id, self.num_rings = self.k.traffic_light.get_control_ids(node_id)
            self.cur_control_id = self.control_id
            # print(node_id, self.control_id, self.num_rings)

            for ring_id in range(0, self.num_rings):
                ring_phases = self.k.traffic_light.get_green_phases(node_id, ring_id)
                self.phases.append(ring_phases)  # get phases index per ring

            for phase_list in self.phases:
                for phase in phase_list:
                    self.observed_phases.append(phase)  # compile all green phases in a list
            print(self.observed_phases)

        # self.current_phase_timings = np.zeros(int(len(self.observed_phases)))
        # reset phase duration
        for node_id in self.target_nodes:
            for phase in self.observed_phases:
                phase_duration, maxd, mind = self.k.traffic_light.get_duration_phase(node_id, phase)
                # self.k.traffic_light.change_phase_duration(node_id, phase, phase_duration)
                self.current_phase_timings.append(phase_duration)
                print('initial phase: {} duration: {} max: {} min: {}'.format(phase, phase_duration, maxd, mind))
            self.rep_name = self.k.traffic_light.get_replication_name(node_id)
            self.cycle = self.k.traffic_light.get_cycle_length(self.node_id, self.control_id)
            print('rep_name: {} cycle_length: {}'.format(self.rep_name, self.cycle))

        self.ignore_policy = False

    @property
    def action_space(self):
        """See class definition."""
        return Tuple(4 * (Discrete(70, ),))

    @property
    def observation_space(self):
        """See class definition."""
        ap = self.additional_params
        shape = ((len(self.target_nodes))*ap['num_incoming_edges_per_node']
                 * (ap['num_detector_types'])*ap['num_measures'])
        return Box(low=0, high=30, shape=(shape, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        min_green_turn = 5
        max_green_turn = 30
        min_green_through = 15

        if self.ignore_policy:
            print('self.ignore_policy is True')
            return

        actions = np.array(rl_actions).flatten()

        turn_minphase_ring1 = min_green_turn if actions[0] <= 5 else actions[0]  # set min green for phase 1
        turn_minphase_ring5 = min_green_turn if actions[2] <= 5 else actions[2]  # set min green for phase 9

        phase_actions = [max_green_turn if actions[0] >= 30 else turn_minphase_ring1,  # Phase 1 (turn)
                         min_green_through if actions[1] <= 15 else actions[1],  # Phase 3 (through)
                         max_green_turn if actions[2] >= 30 else turn_minphase_ring5,  # Phase 5 (turn)
                         min_green_through if actions[3] <= 15 else actions[3], ]  # Phase 7 (through)

        phase_order = [1, 3, 5, 7]

        for phase, action in zip(phase_order, phase_actions):
            if action:
                self.k.traffic_light.change_phase_duration(self.node_id, phase, action, 1)
        self.current_phase_timings = phase_actions

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_detector_types'])
        num_measures = (ap['num_measures'])
        normal = 2000

        # util_per_phase = self.k.traffic_light.get_green_util(self.node_id)
        # print(util_per_phase)
        shape = (num_nodes, num_edges, num_detectors_types, num_measures)
        det_state = np.zeros(shape)
        for i, (node, edge) in enumerate(self.edge_detector_dict.items()):
            for j, (edge_id, detector_info) in enumerate(edge.items()):
                for k, (detector_type, detector_ids) in enumerate(detector_info.items()):
                    if detector_type == 'through':
                        index = (i, j, 0)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)  # mean
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'right':
                        index = (i, j, 1)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)  # mean
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'left':
                        index = (i, j, 2)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'advanced':
                        index = (i, j, 3)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0

        state = det_state.flatten()
        # (state)

        return state

    def compute_reward(self, rl_actions, **kwargs):
        from statistics import mean
        """Computes the sum of queue lengths at all intersections in the network."""
        util_per_phase = self.k.traffic_light.get_green_util(self.node_id)
        reward = 0
        ave_util_section = {568: mean([util_per_phase[0], util_per_phase[1]]),
                            22208: mean([util_per_phase[2], util_per_phase[3]]),
                            400: mean([util_per_phase[0], util_per_phase[1]]),
                            22211: mean([util_per_phase[2], util_per_phase[3]])
                            }
       # queue length as reward

        for section_id in self.past_cumul_queue:
            queue_factor = 1 if ave_util_section.get(section_id) == 0 else 1 + ave_util_section.get(section_id)
            current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)*queue_factor
            queue = current_cumul_queue - self.past_cumul_queue[section_id]
            self.past_cumul_queue[section_id] = current_cumul_queue

            # reward is negative queues
            reward -= (queue**2) * 100
        # note: self.k.simulation.time is flow time
        # f'{slow_time} \t {aimsun_time}
        print(f'{self.k.simulation.time:.0f}', '\t', f'{reward:.4f}', '\t', self.control_id, '\t',
              self.current_phase_timings[0], '\t', self.current_phase_timings[1], '\t', self.current_phase_timings[2], '\t', self.current_phase_timings[3], '\t', self.k.traffic_light.get_intersection_delay(self.node_id))
        # print(self.phase_array)
        # print(self.maxd_list)

        return reward

    def step(self, rl_actions):
        """See parent class."""

        self.time_counter += self.env_params.sims_per_step
        self.step_counter += self.env_params.sims_per_step

        self.apply_rl_actions(rl_actions)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        for _ in range(self.env_params.sims_per_step):
            self.k.simulation.update(reset=False)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon)  # or crash

        # compute the info for each agent
        infos = {}

        self.control_id, self.num_rings = self.k.traffic_light.get_control_ids(self.node_id)
        self.rep_name = self.k.traffic_light.get_replication_name(self.node_id)
        self.cycle = self.k.traffic_light.get_cycle_length(self.node_id, self.control_id)
        # print(self.k.simulation.time,rep_name, self.cycle)

        # compute the reward
        reward = self.compute_reward(rl_actions)

        return next_observation, reward, done, infos

    def reset(self):
        """See parent class.

        The AIMSUN simulation is reset along with other variables.
        """
        # reset the step counter
        self.step_counter = 0

        if self.episode_counter:
            self.k.simulation.reset_simulation()

            episode = self.episode_counter % RLLIB_N_ROLLOUTS

            print('-----------------------')
            print(f'Episode {RLLIB_N_ROLLOUTS if not episode else episode} of {RLLIB_N_ROLLOUTS} complete')
            print('Resetting simulation')
            print('-----------------------')

        # increment episode count
        self.episode_counter += 1

        # reset variables
        # self.current_phase_timings = np.zeros(int(len(self.observed_phases)))
        for section_id in self.past_cumul_queue:
            self.past_cumul_queue[section_id] = 0

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation


class CoordinatedNetwork(Network):
    pass
