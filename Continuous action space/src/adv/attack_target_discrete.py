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

# def pgd(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
#     loss_func = nn.MSELoss()
#     epsilon=attack_config.epsilon_ball
#     agent_inputs = model._build_inputs(batch, t)
#     avail_actions = batch["avail_actions"][:, t]
#     batch_size = batch.batch_size
#     niters = attack_config.attack_niters
#     # print(agent_inputs)
#     step_size = epsilon / niters
    
#     X_adv = Variable(agent_inputs.data, requires_grad=True)
    
#     noise_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon
#     X_adv = agent_inputs.data + noise_0     
#     noise_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
#     X_adv = agent_inputs.data + noise_0

#     X_adv = Variable(X_adv, requires_grad=True)

#     for i in range(niters):
#         logits, hid = model.soft(X_adv, avail_actions, batch, hidden_states, t_ep=t, t_env=t_env, test_mode=True)
#         # loss = loss_func(logits[0], actions[0])
#         loss = - loss_func(logits, actions)
#         opt.zero_grad()
#         loss.backward(retain_graph=True)
#         eta_0 = step_size * X_adv.grad.data.sign()
#         X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
#         eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
#         X_adv.data = agent_inputs.data + eta_0
#     return X_adv.cpu().data.numpy()


def fgsm(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss() # pp

    # loss_func = nn.L1Loss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    # print(agent_inputs)

    X_adv = Variable(agent_inputs.data, requires_grad=True)
    # (self, ep_batch, t, actions=None, test_mode=False):
    logits= model.soft(X_adv, t, test_mode=True, hidden_states = hidden_states)
    # print(logits)
    # aaaaa
    alpha = 0
    # print(tar_actions)
    # print(actions)
    # # aaa
    loss =  alpha*loss_func(logits, actions) - (1-alpha) * loss_func(logits, tar_actions)
    # logits = logits.unsqueeze(0) 
    # loss = -alpha*F.kl_div(actions.log(), logits, reduction='sum') - (1-alpha)*F.kl_div(tar_actions.log(), logits, reduction='sum')
    opt.zero_grad()
    loss.backward(retain_graph=True)

    eta_0 = epsilon * X_adv.grad.data.sign()
    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data - agent_inputs.data, -epsilon, epsilon)
    X_adv.data = agent_inputs.data + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()

# def mifgsm(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):

def pgd(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.MSELoss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    avail_actions = batch["avail_actions"][:, t]
    batch_size = batch.batch_size
 
    niters = attack_config.attack_niters
    # print(niters)
    # aaa
    # print(agent_inputs)
    step_size = epsilon / niters
    alpha = 0.2
    # noise_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon
    # X_adv = agent_inputs.data + noise_0     
    # noise_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    # X_adv = agent_inputs.data + noise_0

    # X_adv = Variable(X_adv, requires_grad=True)
    X_adv = Variable(agent_inputs.data, requires_grad=True)
    

    # optimizer = optim.SGD([X_adv], lr=learning_rate) # model 有待商榷
    # X_adv.data += noise_variance * torch.randn(1, 1)

    for sample in range(niters):
        # print(sample)
        # X_adv.data += noise_variance * torch.randn(1, 1)
        opt.zero_grad()
        logits= torch.FloatTensor(model.soft(X_adv, avail_actions, batch_size, t, hidden_states = hidden_states))
        # print(logits.size())
        # # print(tar_actions)
        # print(actions.size())
        # loss_func(logits.permute(2, 1, 0), actions) 
        # print(logits.dtype)
        # print(tar_actions.dtype)
        # loss = (1-alpha) * loss_func(logits.permute(2, 1, 0), tar_actions.float()) - alpha*loss_func(logits.permute(2, 1, 0), actions.float()) 
        # print(logits.size())
        # print(actions.size())
        # loss_func(logits.permute(0, 2, 1), tar_actions)
        # print('-------------------------------')
        loss = -(1-alpha) * loss_func(logits.permute(0, 2, 1), tar_actions) + alpha*loss_func(logits.permute(0, 2, 1), actions) 
        loss.backward(retain_graph=True)
    
        # Add noise to gradients and perform a gradient step
        # for param in model.parameters():
        #     if param.requires_grad:
        #         param.grad += noise_variance * torch.randn_like(param.grad)
        # optimizer.step()
        # print(X_adv - agent_inputs)
        # aaa
        eta_0 = step_size * X_adv.grad.data.sign()
        X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
        eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
        X_adv.data = agent_inputs.data + eta_0
    
    # print(X_adv - agent_inputs)
    # aaa
    return X_adv.cpu().data.numpy()

def SGLD(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss() # pp

    # loss_func = nn.L1Loss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)

    avail_actions = batch["avail_actions"][:, t]
    batch_size = batch.batch_size
    # print(agent_inputs)
    learning_rate = 0.01#0.01
    noise_variance = 0.#05
    num_samples = attack_config.attack_niters
    alpha = 0.
    step_size = epsilon / num_samples
    
    # noise_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon
    # X_adv = agent_inputs.data + noise_0     
    # noise_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    # X_adv = agent_inputs.data + noise_0

    # X_adv = Variable(X_adv, requires_grad=True)
    X_adv = Variable(agent_inputs.data, requires_grad=True)
    

    optimizer = optim.SGD([X_adv], lr=learning_rate) # model 有待商榷

    for sample in range(num_samples):
        X_adv.data += noise_variance * torch.randn(1, 1)

        optimizer.zero_grad()
        logits= torch.FloatTensor(model.soft(X_adv, avail_actions, batch_size, t, hidden_states = hidden_states))
        # loss = -(1-alpha) * loss_func(logits, tar_actions) + alpha*loss_func(logits, actions) 
        loss = (1-alpha) * loss_func(logits.permute(2, 1, 0), tar_actions.float()) - alpha*loss_func(logits.permute(2, 1, 0), actions.float()) 

        loss.backward(retain_graph=True)
    
        # Add noise to gradients and perform a gradient step
        # for param in model.parameters():
        #     if param.requires_grad:
        #         param.grad += noise_variance * torch.randn_like(param.grad)
        # optimizer.step()
        # print(X_adv - agent_inputs)
        # aaa
        eta_0 = step_size * X_adv.grad.data#.sign()
        # print(eta_0)
        X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
        eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
        X_adv.data = agent_inputs.data + eta_0
    
    # print(X_adv - agent_inputs)
    # aaa
    return X_adv.cpu().data.numpy()

def attack_target_discrete(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,loss_func=nn.CrossEntropyLoss()):

    method = attack_config.attack_target_method
    verbose = attack_config.verbose

    # y = model.soft(obs=X1, agents_available_actions=available_batch)
    if method == "fgsm":
        atk = fgsm
    elif method == "sgld":
        atk = SGLD
    elif method == "pgd":
        atk = pgd
    adv_X = atk(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_states,verbose=verbose)
    return adv_X


