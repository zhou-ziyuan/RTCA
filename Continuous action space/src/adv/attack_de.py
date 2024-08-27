# 最小化的目标函数
# 最大迭代次数
# 离散型变量取值集合的定义 adv_transition_data["avail_actions"] self.args.attack_num
# 根据DE的结果，返回目标智能体与目标动作

# 定义 DE 参数
# pop_size = 50
# mut_rate = 0.8
# cross_rate = 0.7
# max_iter = 100

from .differential_evolution import differential_evolution
import numpy as np
import torch
from torch.autograd import Variable

def perturb_actions(xs, actions, num_attack, pix): # xs:[x, y] x为智能体id y为目标动作  img: 当前的动作
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    # print(xs, actions)
    if xs.ndim < 2:
       xs = np.array([xs])
    batch = len(xs)
    actions = actions.repeat(batch, 1,1)
    # print(actions)
    # xs = xs.astype(int)
    # print(actions)

    count = 0
    # print(xs)
    for x in xs:

        pixels = np.split(x, pix) #len(x)/3
        # print(pixels)
        for pixel in pixels:

            x_pos = int(pixel[0])
            # print(pixel)
            r = pixel[1:]
            # print(x_pos, r)
            # print(r)
            # print(actions[count, x_pos].data)
            actions[count, x_pos] = torch.from_numpy(r).clone()
            # print(actions[count, x_pos].data)
            # aaa
        count += 1
    
    # aaa
	# actions[agents_available_actions == 0] = 10.0
    return actions

def predict_classes(xs, img, learner, agent_inputs, critic_hidden_states, state_batch, batch_size, n_action, agents_available_actions, mix = None, pixels = None): # 扰动后的动作的MIX_Value值
    imgs_perturbed = perturb_actions(xs, img.clone(), n_action, pixels) # 扰动后的动作
    xsbatch = len(xs)
    # print(len(agent_inputs))
    # print(agent_inputs.repeat(xsbatch, 1,1))
    # print(agent_inputs.repeat(xsbatch, 1,1))
    critic_out, critic_hidden_states_ = learner.critic(agent_inputs.repeat(xsbatch, 1,1), imgs_perturbed.detach(),
                                                                critic_hidden_states)
    # print(critic_out.view(batch_size,-1,6).size())
    # print(state_batch.repeat(xsbatch, 1,1).size())
    if mix == "qmix":
        predictions = learner.mixer(critic_out.view(batch_size, -1, 1), state_batch.repeat(xsbatch, 1,1))
    else:
        predictions = learner.mixer(critic_out.view(batch_size,-1,len(agent_inputs)), state_batch.repeat(xsbatch, 1,1))
    predictions = predictions.squeeze().data.cpu().numpy()
    # print(len(predictions))
    # aaa
    return predictions #if minimize else 1 - predictions

def attack_success(x, img, target_calss, learner, agent_inputs, critic_hidden_states, state_batch, batch_size, n_action, verbose=False, mix=None, pix = None): # 看攻击是否成功
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_actions(x, img.clone(), n_action, pix)

    critic_out, critic_hidden_states_ = learner.critic(agent_inputs, attack_image.squeeze().detach(),
                                                                critic_hidden_states)    # print(input)
    if mix == "qmix":
        q_tot = learner.mixer(critic_out.view(batch_size, -1, 1), state_batch).data.cpu().numpy()
    else:
        q_tot = learner.mixer(critic_out.view(batch_size,-1,len(agent_inputs)).squeeze(0), state_batch).data.cpu().numpy()
	# aaa
    # q_tot = learner.mix_value(input, torch.FloatTensor(state_batch)).data.cpu().numpy()[0][0][0]

    if (verbose):
        print ("q_tot: %.4f"%q_tot)
    if (q_tot < target_calss.data.cpu().numpy()[0][0][0]):
        return True # 如果比之前小很多 就算是成功了 阈值的设计？

# img: 动作
# label： 真实的qmix_value
# learner: qmix 网络
# pixels：被攻击智能体的数量
def attack_de(img, label, learner, agent_inputs, critic_hidden_states, n_agent, n_action, state_batch, batch_size,agents_available_actions,
	           pixels=1, maxiter=75, popsize=400, verbose=False, mix = None):

    target_calss = label
    # print(target_calss)
	# print(agents_available_actions)
    # bounds = [[(0,n_agent)] + [(-1, 1)]*n_action] * pixels # len(bounds) = 5
    # print(n_action)
    bounds_ = [(0, n_agent)] + [(-1, 1)] * n_action
    # print(bounds_)
    bounds = bounds_ * pixels

    # bounds1 = [(0,n_agent), (-1, 1)] * pixels
    # print("----------------------------", bounds)
    # print("----------------------------", bounds1)
    popmul = max(1, popsize//len(bounds))
    # print(img)
    # xs, img, learner, action_values, state_batch, agents_available_actions
    predict_fn = lambda xs: predict_classes(
		xs, img, learner, agent_inputs, critic_hidden_states, state_batch, batch_size, n_action, agents_available_actions, mix, pixels) # 要最小化的目标函数
    # x, img, target_calss, learner, action_values, state_batch, verbose=False
    callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, learner, agent_inputs, critic_hidden_states, state_batch, batch_size, n_action, verbose, mix, pixels)
    # callback_fn = None

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    # print(len(bounds))
    # print(inits)
    for init in inits: # 随机初始化
        for i in range(pixels):
            # print(i*(n_action+1)+0)
            init[i*(n_action+1)] = np.random.randint(0, n_agent)
            for j in range(n_action):
                init[i*(n_action+1)+j+1] = (np.random.random() * 2) - 1


        # print(init)
        # sss
        

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
    # print(attack_result)
    # print(img)
    attack_image = perturb_actions(attack_result.x, img.clone(), n_action, pixels)
    # print(attack_image)
    # aaa
    return attack_image[0], attack_result.x#.astype(int)
