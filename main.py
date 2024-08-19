import torch.cuda
from RLFramework.gym import GymnasiumEnvironment
from mlagents_envs.environment import UnityEnvironment
import RLFramework as rl
from FollowbotNets_PPO import *
from MLAgentsEnvironment import MLAgentsEnvironment
# import mlagents.trainers.ppo.trainer as ppo

# ppo.PPOTrainer()
name ="Followbot"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = FollowbotPolicyNet(
    exploration=rl.exp.Gaussian()
).to(device=device)

value = FollowbotValueNet(
    use_target=True,
    tau=0.01,
    policy_net=policy
).to(device=device)

env = MLAgentsEnvironment(behavior_name=name, base_port=5004)#file_name="C:/Users/marks/Documents/Projects/Followbot/build/Followbot.exe")
agent = rl.Agent(policy=policy)

trainer = rl.Trainer(
    agent=agent,
    env=env,
    optimizers=[
        rl.optim.TargetValueOptim(lr=3e-4, epoch=10, batch_size=64, gamma=0.98, level=10),
        rl.optim.ClippedSurrogatePolicyOptim(lr=3e-5, epoch=10, batch_size=64, gamma=0.98, epsilon=0.2,
                                             lamda=0.90, entropy_weight=0.002, use_target_v=True, random_sample=False)
    ],
    # logger=rl.utils.Logger(realtime_plot=True,
    #                        rewards={"reward_sum": "env.episode_reward"}, #, "decay_reward": "env.discount_reward"},
    #                        window_size=1000),
    pi=policy,
    v=value,
    memory=rl.traj.VolatileMemory()
)

version = 0
test = False


def save():
    global version
    version += 1
    if not test:
        trainer.save(f"./saved/{name}_4", version=version)


trainer.add_interval(trainer.train, episode=20, step=500)
trainer.add_interval(value.update_target_network, step=125) # was 50
trainer.add_interval(save, episode=20)

# trainer.load("./saved/Followbot_3", version=version)

if __name__ == '__main__':
    try:
        trainer.run(test_mode=test)

    except KeyboardInterrupt:
        pass

    env.env.close()

    save()

# if __name__ == '__main__':
#     env = UnityEnvironment(base_port=5004)#file_name="C:/Users/marks/Documents/Projects/Followbot/build/Followbot.exe")
#
#     env.reset()
#     behavior_name = list(env.behavior_specs.keys())[0]
#     print(f"name of behavior: {behavior_name}")
#     spec = env.behavior_specs[behavior_name]
#
#     try:
#         for ep in range(10):
#             env.reset()
#
#             decision_steps, terminal_steps = env.get_steps(behavior_name)
#
#             tracked_agent = -1
#             done = False
#             ep_rewards = 0
#
#             step = 0
#
#             # print(list(decision_steps.agent_id))
#
#             while not done:
#                 step += 1
#                 print(f"\rstep: {step}", end="\r")
#                 if tracked_agent == -1 and len(decision_steps) >= 1:
#                     tracked_agent = decision_steps.agent_id[0]
#
#                 old_state = torch.tensor(decision_steps[0][0][0].transpose(2, 0, 1)).to(device)
#
#                 with torch.no_grad:
#                     act = policy(old_state)
#                     actions = [np.random.choice(np.array([0, 1, 2]),
#                                                 p=torch.softmax(p, ).detach().cpu().numpy()) for p in act]
#
#
#                 # print(list(decision_steps[0][0]))
#                 # print(list(decision_steps.values()))
#
#                 action = spec.action_spec.random_action(len(decision_steps))
#
#                 env.set_actions(behavior_name, action)
#
#                 env.step()
#
#                 decision_steps, terminal_steps = env.get_steps(behavior_name)
#                 # print(tracked_agent, list(decision_steps), list(terminal_steps))
#
#                 if tracked_agent in decision_steps:
#                     ep_rewards += decision_steps[tracked_agent].reward
#                 if tracked_agent in terminal_steps:
#                     ep_rewards += terminal_steps[tracked_agent].reward
#                     done = True
#
#             print(f"total reward for ep {ep} is {ep_rewards}")
#
#     except:
#         pass
#
#     env.close()
