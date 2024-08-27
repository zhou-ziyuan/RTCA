from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th
import numpy as np
import copy
import time
import random
from adv.attack import attack_gd
from adv.attack_target import attack_target
from adv.attack_de import attack_de
from adv.attack_q_de import attack_q_de
from adv.attack_q_de_discrete import attack_q_de_discrete
from adv.attack_action import attack_action
from adv.attack_target_discrete import attack_target_discrete
from adv.attack_q_action import attack_q_action
# from learners import REGISTRY as le_REGISTRY
import itertools

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if 'sc2' in self.args.env:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        # self.episode_limit = 2
        # print(self.episode_limit)
        # aa
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
    
    def setup_adv(self, scheme, groups, preprocess, mac, adv_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.adv_mac = adv_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.adv_batch = self.new_batch()
        self.adv_opp_batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, learner = None, learner_q_adv = None,**kwargs):
        self.reset()
        terminated = False
        episode_return = 0
        self.hidden_states = self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.Number_attack > 0 and self.args.attack_method == "adv_tar":
            self.adv_hidden_state = self.adv_mac.init_hidden(batch_size=self.batch_size)
        elif self.args.Number_attack > 0 and self.args.attack_method == "adv_de" or self.args.attack_method == "adv_action" :
            self.critic_hidden_states = learner.critic.init_hidden(self.batch.batch_size)
        env_info = self.env.get_env_info()
        obs_shape = env_info["obs_shape"]
        n_agents = env_info["n_agents"]
        # print(n_agents)
        # aaa
        n_actions = env_info["n_actions"]
        # tt = 0
        while not terminated:
            # print(11111111111111)
            
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                # print('-------------------------------------------')
                # print(self.hidden_states)
                # aaa
                actions, hidden_states_true = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, explore=(not test_mode), hidden_states=self.hidden_states)
            else:
                actions, hidden_states_true = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
            # print(actions)
            # aaaa
            if self.args.Number_attack > 0 :

                if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd_s" or self.args.attack_method == "pgd" or self.args.attack_method == "sgld" or self.args.attack_method == "rand_nosie":
                    # print('-----------------------------')
                    # print(self.mac)
                    adv_inputs_obs = attack_gd(self.mac, self.batch, actions, learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    for i in range (self.args.Number_attack):

                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                elif self.args.attack_method == "adv_tar":
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        tar_actions, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.adv_hidden_state)
                    else:
                        tar_actions, adv_hidden_state_ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.adv_hidden_state)

                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    for i in range (self.args.Number_attack):

                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }

                    self.adv_batch.update(adv_transition_data, ts=self.t)
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t)

                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                    self.adv_hidden_state = adv_hidden_state_



                elif self.args.attack_method == "adv_de":
                    # print(self.mac.soft())
                    agent_inputs = learner._build_inputs(self.batch, self.t)
                    # print(agent_inputs)
                    # print(actions[:, self.t:self.t + 1].detach())
                    critic_out, critic_hidden_states_ = learner.critic(agent_inputs, actions.squeeze().detach(),
                                                                self.critic_hidden_states)
                    
                    mix_value_true = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                    # print(mix_value_true)
                        # chosen_action_qvals = chosen_action_qvals
                    tar_actions, xx = attack_de(img = actions[0], label = mix_value_true, learner = learner, agent_inputs = agent_inputs, critic_hidden_states = self.critic_hidden_states,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1], batch_size=self.batch.batch_size,\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False, mix = self.args.mixer)
                    
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    for i in range (self.args.Number_attack):
                        # print((n_actions+1)*i)
                        adv_inputs[int(xx[(n_actions+1)*i])] = adv_inputs_obs[int(xx[(n_actions+1)*i])]


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                    self.critic_hidden_states = critic_hidden_states_

                elif self.args.attack_method == "adv_action":
                    # print(self.mac.soft())
                    # attack_action(model, batch, actions, learner, mix_value_true, critic_hidden_states, attack_config, t, loss_func=nn.CrossEntropyLoss()):
                    agent_inputs = learner._build_inputs(self.batch, self.t)
                    # print(agent_inputs)
                    # print(actions[:, self.t:self.t + 1].detach())
                    critic_out, critic_hidden_states_ = learner.critic(agent_inputs, actions.squeeze().detach(),
                                                                self.critic_hidden_states)
    
                    mix_value_true = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                    # print(mix_value_true)
                        # chosen_action_qvals = chosen_action_qvals

                    # attack_action(model, batch, actions, learner, mix_value_true, critic_hidden_states, attack_config, t, loss_func=nn.CrossEntropyLoss()):
                    tar_actions = attack_action(self.mac, self.batch, agent_inputs,actions, learner, mix_value_true, self.critic_hidden_states, self.args, self.t)
                    
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    all_combinations_indices = list(itertools.combinations(range(0, n_agents), self.args.Number_attack))
                    min_y_value = float('inf')
                    optimal_indices = None
                    for indices  in all_combinations_indices:
                        replaced_actions = actions.detach()
                        for i, index in enumerate(indices):
                            # print(indices)
                            # print(replaced_actions)
                            # print(tar_actions)
                            replaced_actions[0][index] =  tar_actions[0][index]
                        critic_out, critic_hidden_states_ = learner.critic(agent_inputs, replaced_actions.squeeze().detach(),
                                                                self.critic_hidden_states)

                        y_value = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                        if y_value < min_y_value:
                            min_y_value = y_value
                            optimal_indices = indices

                    for xx in optimal_indices:
                        adv_inputs[xx] = adv_inputs_obs[xx].copy()


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                    self.critic_hidden_states = critic_hidden_states_

                elif self.args.attack_method == "adv_q_action":
                    # print(self.mac.soft())
                    # attack_action(model, batch, actions, learner, mix_value_true, critic_hidden_states, attack_config, t, loss_func=nn.CrossEntropyLoss()):
                    mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    # print(actions.view(-1).unsqueeze(0))
                    tar_actions = attack_q_action(self.mac, self.batch, actions, learner_q_adv, mix_value_true, self.args, self.t)
                    
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    # attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    # for i in range (self.args.Number_attack):

                    #     adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()
                    all_combinations_indices = list(itertools.combinations(range(0, n_agents), self.args.Number_attack))
                    min_y_value = float('inf')
                    optimal_indices = None
                    for indices  in all_combinations_indices:
                        replaced_actions = actions.detach()
                        for i, index in enumerate(indices):
                            # print(indices)
                            # print(replaced_actions)
                            # print(tar_actions)
                            replaced_actions[0][index] =  tar_actions[0][index]

                        y_value = learner_q_adv.mix_value(replaced_actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                        if y_value < min_y_value:
                            min_y_value = y_value
                            optimal_indices = indices

                    for xx in optimal_indices:
                        adv_inputs[xx] = adv_inputs_obs[xx].copy()


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                    # self.critic_hidden_states = critic_hidden_states_

                elif (self.args.attack_method == "q_adv_de" ) and test_mode == True:
                    # print(actions[0].to("cpu"))
                    # print(th.FloatTensor(pre_transition_data["state"]))
                    # print(actions)
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions = th.argmax(actions, dim=-1).long()
                        mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                        # print('---------------------------------')
                        tar_actions, xx = attack_q_de_discrete(img = actions[0], label = mix_value_true, learner=learner_q_adv, actions=actions,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1],
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)
                        # tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner = learner, actions=actions,\
                                                    # n_agent = n_agents, n_action= n_actions, state_batch= pre_transition_data["state"],\
                                                    # agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    # maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                        adv_inputs_obs = attack_target_discrete(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                        
                        adv_inputs = pre_transition_data["obs"][0].copy()
                        for i in range (self.args.Number_attack):
                        # print(adv_inputs[0])
                            adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]
                    else:
                        # print(11111)
                        mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    
                        tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner=learner_q_adv,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1], batch_size=self.batch.batch_size,\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=200, verbose=False)
                    
                        adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                        # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                        # print(111111111)
                        adv_inputs = pre_transition_data["obs"][0].copy()
                        # print(adv_inputs)    
                        # print(xx)
                        for i in range (self.args.Number_attack):
                        # print((n_actions+1)*i)
                            adv_inputs[int(xx[(n_actions+1)*i])] = adv_inputs_obs[int(xx[(n_actions+1)*i])]


                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                    else:
                        actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                    self.hidden_states = hidden_states_
                    # self.critic_hidden_states = critic_hidden_states_
                    

            else:

                self.hidden_states = hidden_states_true


            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = th.argmax(actions, dim=-1).long()

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                # print(reward)
                # print(reward)
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
                
                # aaa
                episode_return += reward

            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
                
                episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }


            self.batch.update(post_transition_data, ts=self.t)
            if self.args.Number_attack > 0 :
                if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd_s" or self.args.attack_method == "pgd" or self.args.attack_method == "sgld" or self.args.attack_method == "rand_nosie" or self.args.attack_method == "adv_de" or self.args.attack_method == "adv_action" or self.args.attack_method == "q_adv_de" or self.args.attack_method == "adv_q_action":
                    self.adv_batch.update(post_transition_data, ts=self.t)
                elif self.args.attack_method == "adv_tar":
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    opp_post_transition_data = {
                        "actions": actions,
                        "reward": [(-reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        # Select actions in the last stored state
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions, hidden_states_ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode), hidden_states=self.hidden_states)
        else:
            actions, hidden_states_ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)


        if self.args.Number_attack > 0:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd_s" or self.args.attack_method == "pgd" or self.args.attack_method == "sgld" or self.args.attack_method == "rand_nosie":
                adv_inputs_obs = attack_gd(self.mac, self.batch, actions, learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                # print(111111111)
                adv_inputs = last_data["obs"][0].copy()
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                # print(adv_inputs)    
                for i in range (self.args.Number_attack):

                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()


                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
            
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                # self.adv_batch.update(last_data, ts=self.t)
                self.adv_batch.update({"actions": actions}, ts=self.t)

            elif self.args.attack_method == "adv_tar":
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                        tar_actions, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.adv_hidden_state)
                else:
                    tar_actions, adv_hidden_state_ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.adv_hidden_state)

                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                
                adv_inputs = last_data["obs"][0].copy()
  
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                # print(adv_inputs)    
                for i in range (self.args.Number_attack):

                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()


                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()                
                self.adv_batch.update({"actions": actions}, ts=self.t)
                self.adv_opp_batch.update({"actions": tar_actions}, ts=self.t)


            elif self.args.attack_method == "adv_de" :

                agent_inputs = learner._build_inputs(self.batch, self.t)
                # print(agent_inputs)
                # print(actions[:, self.t:self.t + 1].detach())
                critic_out, critic_hidden_states_ = learner.critic(agent_inputs, actions.squeeze().detach(),
                                                                self.critic_hidden_states)
                mix_value_true = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                    
                tar_actions, xx = attack_de(img = actions[0], label = mix_value_true, learner = learner, agent_inputs = agent_inputs, critic_hidden_states = self.critic_hidden_states,\
                                                    n_agent = self.args.Number_attack, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1], batch_size=self.batch.batch_size,\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False, mix = self.args.mixer)
                    
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                # print(111111111)
                adv_inputs = last_data["obs"][0].copy()
                    # print(adv_inputs)    
                for i in range (self.args.Number_attack):

                    adv_inputs[int(xx[(n_actions+1)*i])] = adv_inputs_obs[int(xx[(n_actions+1)*i])]


                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
            
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                # self.adv_batch.update(last_data, ts=self.t)
                self.adv_batch.update({"actions": actions}, ts=self.t)

            elif self.args.attack_method == "adv_action" :

                agent_inputs = learner._build_inputs(self.batch, self.t)
                # print(agent_inputs)
                # print(actions[:, self.t:self.t + 1].detach())
                critic_out, critic_hidden_states_ = learner.critic(agent_inputs, actions.squeeze().detach(),
                                                                self.critic_hidden_states)
                mix_value_true = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                    
                tar_actions = attack_action(self.mac, self.batch, agent_inputs,actions, learner, mix_value_true, self.critic_hidden_states, self.args, self.t)
                    
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                # print(111111111)
                adv_inputs = last_data["obs"][0].copy()
                    # print(adv_inputs)    
                # attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                # for i in range (self.args.Number_attack):

                #     adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()

                all_combinations_indices = list(itertools.combinations(range(0, n_agents), self.args.Number_attack))
                min_y_value = float('inf')
                optimal_indices = None
                for indices  in all_combinations_indices:
                    replaced_actions = actions.detach()
                    for i, index in enumerate(indices):
                        # print(indices)
                        # print(replaced_actions)
                        # print(tar_actions)
                        replaced_actions[0][index] =  tar_actions[0][index]

                    critic_out, critic_hidden_states_ = learner.critic(agent_inputs, replaced_actions.squeeze().detach(),
                                                                self.critic_hidden_states)

                    y_value = learner.mixer(critic_out.view(self.batch.batch_size, -1, 1), self.batch["state"][:, self.t:self.t + 1])
                    if y_value < min_y_value:
                        min_y_value = y_value
                        optimal_indices = indices

                for xx in optimal_indices:
                    adv_inputs[xx] = adv_inputs_obs[xx].copy()

                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
            
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                # self.adv_batch.update(last_data, ts=self.t)
                self.adv_batch.update({"actions": actions}, ts=self.t)

            elif self.args.attack_method == "adv_q_action" :

                # agent_inputs = learner._build_inputs(self.batch, self.t)
                # print(agent_inputs)
                # print(actions[:, self.t:self.t + 1].detach())
                # critic_out, critic_hidden_states_ = learner.critic(agent_inputs, actions.squeeze().detach(),
                #                                                 self.critic_hidden_states)
                mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    # print(actions.view(-1).unsqueeze(0))
                tar_actions = attack_q_action(self.mac, self.batch, actions, learner_q_adv, mix_value_true, self.args, self.t)
                    
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                # print(111111111)
                adv_inputs = last_data["obs"][0].copy()
                    # print(adv_inputs)    
                # attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)

                all_combinations_indices = list(itertools.combinations(range(0, n_agents), self.args.Number_attack))
                min_y_value = float('inf')
                optimal_indices = None
                for indices  in all_combinations_indices:
                    replaced_actions = actions.detach()
                    for i, index in enumerate(indices):
                        # print(indices)
                        # print(replaced_actions)
                        # print(tar_actions)
                        replaced_actions[0][index] =  tar_actions[0][index]

                    y_value = learner_q_adv.mix_value(replaced_actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    if y_value < min_y_value:
                        min_y_value = y_value
                        optimal_indices = indices

                for xx in optimal_indices:
                    adv_inputs[xx] = adv_inputs_obs[xx].copy()


                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
            
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)

                # self.adv_batch.update(last_data, ts=self.t)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                self.adv_batch.update({"actions": actions}, ts=self.t)

            elif self.args.attack_method == "q_adv_de":
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                    mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    tar_actions, xx = attack_q_de_discrete(img = actions[0], label = mix_value_true, learner=learner_q_adv, actions=actions,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1],
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)
                    # tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner = learner, actions=actions,\
                                                    # n_agent = n_agents, n_action= n_actions, state_batch= pre_transition_data["state"],\
                                                    # agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    # maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                    adv_inputs_obs = attack_target_discrete(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    for i in range (self.args.Number_attack):
                        # print(adv_inputs[0])
                        adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]
                else:
                    mix_value_true = learner_q_adv.mix_value(actions.view(-1).unsqueeze(0), th.FloatTensor(pre_transition_data["state"]))
                    
                    tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner=learner_q_adv,\
                                                    n_agent = self.args.Number_attack, n_action= n_actions, state_batch= self.batch["state"][:, self.t:self.t + 1], batch_size=self.batch.batch_size,\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=200, verbose=False)
                    
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions,learner.agent_optimiser, self.args, self.t, self.t_env, self.hidden_states)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    # print(111111111)
                    adv_inputs = last_data["obs"][0].copy()
                    # print(adv_inputs)    
                    for i in range (self.args.Number_attack):

                        adv_inputs[int(xx[(n_actions+1)*i])] = adv_inputs_obs[int(xx[(n_actions+1)*i])]


                adv_last_data = {
                        "state": last_data["state"],
                        "avail_actions": last_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                self.adv_batch.update(adv_last_data, ts=self.t)
            
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode), hidden_states=self.hidden_states)
                else:
                    actions, hidden_states_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, hidden_states=self.hidden_states)

                # self.adv_batch.update(last_data, ts=self.t)
                if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                    actions = th.argmax(actions, dim=-1).long()
                self.adv_batch.update({"actions": actions}, ts=self.t)

            
        else:
            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = th.argmax(actions, dim=-1).long()
            self.batch.update({"actions": actions}, ts=self.t)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
        # print(episode_return)

        cur_returns.append(episode_return)
        if self.args.evaluate:
            print(episode_return,'-------------', np.mean(cur_returns), np.std(cur_returns))
            if 'sc2' in self.args.env:
                print(episode_return,'-------------', cur_stats["battle_won"])
                print(self.t)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode-1):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        
        if test_mode:
            return self.t

        if self.args.Number_attack > 0:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd_s" or self.args.attack_method == "pgd" or self.args.attack_method == "sgld" or self.args.attack_method == "rand_nosie" or self.args.attack_method == "adv_de" or self.args.attack_method == "adv_action" or self.args.attack_method == "q_adv_de" or self.args.attack_method == "adv_q_action":
                return self.adv_batch
            elif self.args.attack_method == "adv_tar":
                return self.adv_batch, self.adv_opp_batch
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def find_minimal_combination(data, n, y_function):
        m = len(data)
    
        if n > m:
            raise ValueError("n should be less than or equal to the number of data points (m).")

        # 生成所有可能的组合
        all_combinations = list(itertools.combinations(data, n))

        # 初始化最小值和对应的组合
        min_y_value = float('inf')
        optimal_combination = None

        # 遍历所有组合，计算 y 函数的值，并更新最小值和对应组合
        for combination in all_combinations:
            y_value = y_function(combination)
            if y_value < min_y_value:
                min_y_value = y_value
                optimal_combination = combination

        return optimal_combination, min_y_value