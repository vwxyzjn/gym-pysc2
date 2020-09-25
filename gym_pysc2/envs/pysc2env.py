# reference: https://github.com/inoryy/reaver/blob/master/reaver/envs/sc2.py
import numpy as np
import gym
import pygame
from gym import error, spaces, utils
from gym.utils import seeding
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env.environment import StepType

class PySC2Env(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(
        self,
        action_ids=None,
        spatial_dim=16,
        step_mul=8,
        map_name="MoveToBeacon",
        visualize=True) -> None:

        super().__init__()
        self.action_ids = action_ids
        self.spatial_dim = spatial_dim
        self.step_mul = step_mul
        self.map_name = map_name
        self.visualize = visualize
        
        # preprocess
        if not self.action_ids:
            self.action_ids = [f.id for f in actions.FUNCTIONS]
        self.reverse_action_ids = np.zeros(max(self.action_ids)+1, dtype=np.int16)
        for idx, aid in enumerate(self.action_ids):
            self.reverse_action_ids[aid] = idx

        # setup env
        self._env = sc2_env.SC2Env(
            map_name=self.map_name,
            visualize=self.visualize,
            agent_interface_format=[sc2_env.parse_agent_interface_format(
                feature_screen=self.spatial_dim,
                feature_minimap=self.spatial_dim,
                rgb_screen=None,
                rgb_minimap=None
            )],
            step_mul=self.step_mul,
            players=[sc2_env.Agent(sc2_env.Race.terran)])
        self.temp_obs = self._env.reset()[0].observation

        # observation space
        obs_features = {
            'screen': ['player_relative', 'selected', 'visibility_map', 'unit_hit_points_ratio', 'unit_density'],
            'minimap': ['player_relative', 'selected', 'visibility_map', 'camera'],
            'non-spatial': ['available_actions', 'player']}
        screen_feature_to_idx = {feat: idx for idx, feat in enumerate(features.SCREEN_FEATURES._fields)}
        minimap_feature_to_idx = {feat: idx for idx, feat in enumerate(features.MINIMAP_FEATURES._fields)}
        self.feature_masks = {
            'screen': [screen_feature_to_idx[f] for f in obs_features['screen']],
            'minimap': [minimap_feature_to_idx[f] for f in obs_features['minimap']]
        }

        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(len(obs_features['screen'])*spatial_dim*spatial_dim+ \
            len(obs_features['minimap'])*spatial_dim*spatial_dim+ \
            len(self.action_ids)+len(self.temp_obs["player"]),)
        )
        self.feature_flatten_shapes = (len(obs_features['screen'])*spatial_dim*spatial_dim,) + \
            (len(obs_features['minimap'])*spatial_dim*spatial_dim,) + \
            (len(self.action_ids)+len(self.temp_obs["player"]),)
        self.feature_original_shapes = [
            (len(obs_features['screen']), spatial_dim, spatial_dim),
            (len(obs_features['minimap']), spatial_dim, spatial_dim),
            (len(self.action_ids)+len(self.temp_obs["player"]),)
        ]

        # action space
        self.args = [
            'screen',
            'minimap',
            'screen2',
            'queued',
            'control_group_act',
            'control_group_id',
            'select_add',
            'select_point_act',
            'select_unit_act',
            # 'select_unit_id'
            'select_worker',
            'build_queue_id',
            # 'unload_id'
        ]

        self.args_idx = {}
        action_args = ()
        for arg_name in self.args:
            arg = getattr(self._env.action_spec()[0][0], arg_name)
            self.args_idx[arg_name] = slice(len(action_args)+1, len(action_args)+1+len(arg.sizes))
            action_args += arg.sizes
        self.action_space = spaces.MultiDiscrete([
            len(self.action_ids),
        ] + list(action_args))

    def step(self, action):
        defaults = {
            'control_group_act': 0,
            'control_group_id': 0,
            'select_point_act': 0,
            'select_unit_act': 0,
            'select_unit_id': 0,
            'build_queue_id': 0,
            'unload_id': 0,
        }
        action_id_idx, args = action[0], []
        action_id = self.action_ids[action_id_idx]
        for arg_type in actions.FUNCTIONS[action_id].args:
            arg_name = arg_type.name
            if arg_name in self.args:
                arg = action[self.args_idx[arg_name]]
                # # arg = action[self.args.index(arg_name)]
                # # pysc2 expects all args in their separate lists
                # if type(arg) not in [list, tuple]:
                #     arg = [arg]
                # # pysc2 expects spatial coords, but we have flattened => attempt to fix
                # # if len(arg_type.sizes) > 1 and len(arg) == 1:
                # #     arg = [arg[0] % self.spatial_dim, arg[0] // self.spatial_dim]
                args.append(arg)
            else:
                args.append([defaults[arg_name]])

        response = self._env.step([actions.FunctionCall(action_id, args)])[0]
        raw_obs = response.observation

        # action masking
        action_id_mask = np.zeros(len(self.action_ids))
        for available_action_id in raw_obs['available_actions']:
            action_id_mask[self.reverse_action_ids[available_action_id]] = 1
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[:len(self.action_ids)] = action_id_mask
        self.available_actions = raw_obs['available_actions']

        return np.concatenate([
            raw_obs["feature_screen"][self.feature_masks["screen"]].flatten(),
            raw_obs["feature_minimap"][self.feature_masks["minimap"]].flatten(),
            np.zeros(len(self.action_ids)+len(self.temp_obs["player"]))
        ]), response.reward, response.step_type == StepType.LAST, {}

    def reset(self):
        response = self._env.reset()[0]
        raw_obs = response.observation
        
        # action masking
        action_id_mask = np.zeros(len(self.action_ids))
        for available_action_id in raw_obs['available_actions']:
            action_id_mask[self.reverse_action_ids[available_action_id]] = 1
        self.action_mask = np.ones(self.action_space.nvec.sum())
        self.action_mask[:len(self.action_ids)] = action_id_mask
        self.available_actions = raw_obs['available_actions']

        return np.concatenate([
            raw_obs["feature_screen"][self.feature_masks["screen"]].flatten(),
            raw_obs["feature_minimap"][self.feature_masks["minimap"]].flatten(),
            np.zeros(len(self.action_ids)+len(self.temp_obs["player"]))
        ])

    def render(self, mode="human"):
        if mode == "rgb_array":
            x = self._env._renderer_human._window.copy()
            array = pygame.surfarray.pixels3d(x)
            array = np.transpose(array, axes=(1, 0, 2))
            del x
            return array

    # def step(self, action):
    #     defaults = {
    #         'control_group_act': 0,
    #         'control_group_id': 0,
    #         'select_point_act': 0,
    #         'select_unit_act': 0,
    #         'select_unit_id': 0,
    #         'build_queue_id': 0,
    #         'unload_id': 0,
    #     }
    #     action_id_idx, args = action[0], []
    #     action_id = self.action_ids[action_id_idx]
    #     for arg_type in actions.FUNCTIONS[action_id].args:
    #         arg_name = arg_type.name
    #         if arg_name in self.args:
    #             arg = action[self.args_idx[arg_name]]
    #             # # arg = action[self.args.index(arg_name)]
    #             # # pysc2 expects all args in their separate lists
    #             # if type(arg) not in [list, tuple]:
    #             #     arg = [arg]
    #             # # pysc2 expects spatial coords, but we have flattened => attempt to fix
    #             # # if len(arg_type.sizes) > 1 and len(arg) == 1:
    #             # #     arg = [arg[0] % self.spatial_dim, arg[0] // self.spatial_dim]
    #             args.append(arg)
    #         else:
    #             args.append([defaults[arg_name]])

    #     response = self._env.step([actions.FunctionCall(action_id, args)])[0]
    #     raw_obs = response.observation

    #     # action masking
    #     action_id_mask = np.zeros(len(self.action_ids))
    #     for available_action_id in raw_obs['available_actions']:
    #         action_id_mask[self.reverse_action_ids[available_action_id]] = 1
    #     self.action_mask = np.zeros(self.action_space.nvec.sum())
    #     self.action_mask[:len(self.action_ids)] = action_id_mask
    #     self.action_mask[len(self.action_ids):len(self.action_ids)+2] = 1
    #     self.available_actions = raw_obs['available_actions']

    #     return np.concatenate([
    #         raw_obs["feature_screen"][self.feature_masks["screen"]].flatten(),
    #         raw_obs["feature_minimap"][self.feature_masks["minimap"]].flatten(),
    #         np.zeros(len(self.action_ids)+len(self.temp_obs["player"]))
    #     ]), response.reward, response.step_type == StepType.LAST, {}

    # def reset(self):
    #     response = self._env.reset()[0]
    #     raw_obs = response.observation
        
    #     # action masking
    #     action_id_mask = np.zeros(len(self.action_ids))
    #     for available_action_id in raw_obs['available_actions']:
    #         action_id_mask[self.reverse_action_ids[available_action_id]] = 1
    #     self.action_mask = np.zeros(self.action_space.nvec.sum())
    #     self.action_mask[:len(self.action_ids)] = action_id_mask
    #     self.action_mask[len(self.action_ids):len(self.action_ids)+2] = 1
    #     self.available_actions = raw_obs['available_actions']

    #     return np.concatenate([
    #         raw_obs["feature_screen"][self.feature_masks["screen"]].flatten(),
    #         raw_obs["feature_minimap"][self.feature_masks["minimap"]].flatten(),
    #         np.zeros(len(self.action_ids)+len(self.temp_obs["player"]))
    #     ])
