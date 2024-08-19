import numpy as np
import RLFramework as rl
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment


class MLAgentsEnvironment(rl.Environment):
    def __init__(self, *args,  behavior_name: str, **kwargs):
        super().__init__(
            # observation_space=rl.space.Continuous(
            #   upper=np.ones(10),
            #   lower=-np.ones(10)
            # ),
            observation_space=rl.space.Continuous(
                upper=np.zeros((3,224,224)),
                lower=np.ones((3,224,224))
            ),
            action_space=rl.space.Continuous(
                upper=np.ones(2),
                lower=-np.ones(2)
            )
        )

        self.episode_count = 0
        self.behavior_name = behavior_name
        self.env = UnityEnvironment(*args, **kwargs)

        self.env.reset()
        behavior_name = list(self.env.behavior_specs.keys())[0]
        print(f"name of behavior: {behavior_name}")
        self.behavior_name = behavior_name
        spec = self.env.behavior_specs[behavior_name]

        self.unity_reward = 0
        self.episode_reward = 0

        self.reset_params()

    def update(self, state, action):
        # print(action)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.array([action]))

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.unity_reward = decision_steps[0].reward
        obs = decision_steps[0].obs[0].transpose(2,0,1)
        end = 0 in terminal_steps

        # print(obs)

        return obs, end

    def reward(self, state, action, next_state):
        self.episode_reward += self.unity_reward
        return self.unity_reward

    def reset_params(self):
        self.episode_count += 1
        print(f"episode {self.episode_count}: reward {self.episode_reward}")

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        obs = decision_steps[0].obs[0].transpose(2, 0, 1)

        self.init_state(obs)

        self.unity_reward = 0
        self.episode_reward = 0
        self.env.reset()
