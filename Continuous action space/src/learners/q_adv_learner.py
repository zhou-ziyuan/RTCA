import copy
from components.episode_buffer import EpisodeBatch
from adversarial_modules.q_evaluate import QCritic
from adversarial_modules.q_evaluate_cnn import QCritic_CNN
import torch as th
from torch.optim import RMSprop
from torch.distributions import Categorical

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        # self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        if args.fc == "CNN":
            self.mixer = QCritic_CNN(args)
        else:
            self.mixer = QCritic(args)

        self.params = list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.hidden_state = self.mac.init_hidden(batch.batch_size)

        target_mac_actions = []
        
        self.target_hidden_state = self.target_mac.init_hidden(batch.batch_size)

        if self.mixer is not None:
            # print(batch["state"][:, :-1].size())
            size = batch["actions"][:, :-1].squeeze(-1).size()
            tar_size = batch["actions"][:,1 :].squeeze(-1).size()
            # print(batch["actions"][:, :-1].squeeze(-1).view(size[0],size[1],size[2]*size[3]).size())
            # chosen_action_qvals = self.mixer(batch["actions"][:, :-1].squeeze(-1).view(size[0],size[1],size[2]*size[3]), batch["state"][:, :-1])
            # target_max_qvals = self.target_mixer(batch["actions"][:,1 :].squeeze(-1).view(tar_size[0],tar_size[1],tar_size[2]*tar_size[3]), batch["state"][:, 1:])
            # print(batch["state"][:, :-1].size())
            # print(batch["state"][:, :-1].reshape(size[0]*size[1],1,-1))
            chosen_action_qvals = self.mixer(batch["actions"][:, :-1].squeeze(-1).reshape(size[0]*size[1],1,size[2]*size[3]), batch["state"][:, :-1].reshape(size[0]*size[1],1,-1))
            target_max_qvals = self.target_mixer(batch["actions"][:,1 :].squeeze(-1).reshape(tar_size[0]*tar_size[1],1,tar_size[2]*tar_size[3]), batch["state"][:, 1:].reshape(tar_size[0]*tar_size[1],1,-1))
        # print(target_max_qvals.size())
            chosen_action_qvals = chosen_action_qvals.reshape(size[0], size[1], -1)
            target_max_qvals = target_max_qvals.reshape(tar_size[0], tar_size[1], -1)
        # print(rewards.size())
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    
    def load_q_adv_models(self, path):
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    
    def mix_value(self, chosen_actions, state):
        
        if self.mixer is not None:
            # print(chosen_actions.size())
            # new_tensor_actions = th.zeros((1, 25, chosen_actions.size()[2]))
            # new_tensor_actions[:, 0, :] = chosen_actions
            # new_tensor_state = th.zeros((1, 25, state.size()[2]))
            # new_tensor_state[:, 0, :] = chosen_actions
            chosen_action_qvals = self.mixer(chosen_actions, state)
        else:
            chosen_action_qvals = None

        return chosen_action_qvals


        