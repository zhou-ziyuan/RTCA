from pickle import TRUE
from random import random
import torch
from torch import autograd
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
TARGET_MULT = 10000.0
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

def pgd(model, batch, agent_inputs ,actions, learner, mix_value_true, t, critic_hidden_states,verbose=False,  env_id=""):
    loss_func = nn.MSELoss()
    epsilon=1
    # agent_inputs = model._build_inputs(batch, t)
    X_adv = Variable(actions.data, requires_grad=True)
    # print(agent_inputs)
    step_size = epsilon / 20
    
    X_adv = Variable(agent_inputs.data, requires_grad=True)
    
    noise_0 = 2 * epsilon * torch.rand(actions.size()) - epsilon
    X_adv = actions.data + noise_0     
    noise_0 = torch.clamp(X_adv.data- actions.data, -epsilon, epsilon)
    X_adv = actions.data + noise_0

    X_adv = Variable(X_adv, requires_grad=True)

    for i in range(20):
        logits,_= learner.critic(agent_inputs, X_adv.squeeze(), critic_hidden_states)
        mix_value = learner.mixer(logits.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
        pg_loss = -mix_value.mean()
        learner.agent_optimiser.zero_grad()
        pg_loss.backward(retain_graph=True)
        eta_0 = epsilon * X_adv.grad.data.sign()
        X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
        X_adv.data = torch.clamp(X_adv.data, -epsilon, epsilon)
        # X_adv.data = actions.data + eta_0
    return X_adv#.cpu().data.numpy()


def fgsm(model, batch, agent_inputs ,actions, learner, mix_value_true, t, critic_hidden_states,verbose=False,  env_id=""):
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()
    epsilon=1
    # agent_inputs = model._build_inputs(batch, t)
    # print(agent_inputs)

    X_adv = Variable(actions.data, requires_grad=True)
    # print(X_adv.grad)
    logits,_= learner.critic(agent_inputs, X_adv.squeeze(), critic_hidden_states)
    mix_value = learner.mixer(logits.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
    pg_loss = -mix_value.mean() #+ (pi**2).mean() * 1e-3
    learner.agent_optimiser.zero_grad()
    # print(pg_loss)
    pg_loss.backward(retain_graph=True)

    eta_0 = epsilon * X_adv.grad.data.sign()
    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data - actions.data, -epsilon, epsilon)
    X_adv.data = actions.data + eta_0
    # print(X_adv)
    # print(actions)
    return X_adv#.cpu().data.numpy()

def SGLD(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss() # pp

    # loss_func = nn.L1Loss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    # print(agent_inputs)
    learning_rate = 0.01
    noise_variance = 0.1
    num_samples = attack_config.attack_niters
    alpha = 1
    step_size = epsilon / num_samples
    X_adv = Variable(agent_inputs.data, requires_grad=True)
    optimizer = optim.SGD([X_adv], lr=learning_rate) # model 有待商榷 model.parameters()
    # optimizer.param_groups[0]['params'][0].data += noise_variance * torch.randn(1, 1)

    for sample in range(num_samples):
        optimizer.zero_grad()
        logits= model.soft(X_adv, t, test_mode=True, hidden_states = hidden_states)
        loss = alpha*loss_func(logits, actions) #+ (1-alpha) * loss_func(logits, tar_actions)
        loss.backward(retain_graph=True)
    
        # Add noise to gradients and perform a gradient step
        # for param in X_adv.parameters():
        #     if param.requires_grad:
        #         param.grad += noise_variance * torch.randn_like(param.grad)
        # optimizer.step()
        eta_0 = step_size * X_adv.grad.data#.sign()
        X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
        eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
        X_adv.data = agent_inputs.data + eta_0
    
    # print(X_adv - agent_inputs)
    # aaa
    return X_adv.cpu().data.numpy()

def rand_nosie(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    device = agent_inputs.device
    # X_adv = Variable(agent_inputs.data, requires_grad=True)
    X_adv = agent_inputs.clone().detach().to(device)
    X_adv.requires_grad = True
    eta_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon
    eta_0 = eta_0.to(device)
    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    X_adv.data = agent_inputs.data + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()


def attack_action(model, batch, agent_input, actions, learner, mix_value_true, critic_hidden_states, attack_config, t, loss_func=nn.CrossEntropyLoss()):

    method = attack_config.attack_target_method
    verbose = attack_config.verbose

    # y = model.soft(obs=X1, agents_available_actions=available_batch)
    # if method == 'fgsm':
    atk = pgd

    adv_X = atk(model, batch, agent_input, actions, learner, mix_value_true, t, critic_hidden_states,verbose=False,  env_id="")
    return adv_X


