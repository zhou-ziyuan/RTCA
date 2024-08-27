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

def perturb_actions(xs, actions): # xs:[x, y] x为智能体id y为目标动作  img: 当前的动作
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
       xs = np.array([xs])
    batch = len(xs)
    actions = actions.repeat(batch, 1,1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x)/2)
        # print(pixels)
        for pixel in pixels:
            x_pos, r = pixel
            # print(x_pos, r)
            # print(actions[count])
            actions[count, 0 ,x_pos] = r
        count += 1
	# actions[agents_available_actions == 0] = 10.0
    return actions

def predict_classes(xs, img, learner, action_values, state_batch, agents_available_actions): # 扰动后的动作的MIX_Value值
    imgs_perturbed = perturb_actions(xs, img.clone())

    action_values[agents_available_actions == 0] = 10000000.0
    action_values = action_values.repeat(len(xs), 1,1)
	
    value_perturbed = torch.gather(action_values, dim=2, index=imgs_perturbed).squeeze(0)#.view(-1,n_actor, n_agents, )

    input = Variable(value_perturbed, volatile=True)#.no_grad()#.cuda()
	# print(value_perturbed.size())
	# print(torch.FloatTensor(state_batch).repeat(len(xs), 1).size())
    predictions = learner.mix_value(input, torch.FloatTensor(state_batch).repeat(len(xs), 1))
    predictions = predictions.data.cpu().numpy()
    return predictions #if minimize else 1 - predictions

def attack_success(x, img, target_calss, learner, action_values, state_batch, verbose=False): # 看攻击是否成功
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_actions(x, img.clone())
    # chosen_action_qvals = torch.gather(action_values[0], dim=1, index=np.transpose(actions))
    # print(attack_image)
    # print(np.transpose(attack_image[0]))
    # print(action_values)
    input = torch.gather(action_values[0], dim=1, index=np.transpose(attack_image[0]))#.squeeze(3)#.view(-1,n_actor, n_agents)
    # print(input)
    size_0 = input.size(0)
    input = input.view(1, size_0).unsqueeze(0)
    input = Variable(input, volatile=True)
    
    
	# aaa
    q_tot = learner.mix_value(input, torch.FloatTensor(state_batch)).data.cpu().numpy()[0][0][0]

    if (verbose):
        print ("q_tot: %.4f"%q_tot)
    if (q_tot < target_calss.data.cpu().numpy()[0][0][0]):
        return True # 如果比之前小很多 就算是成功了 阈值的设计？

# img: 动作
# label： 真实的qmix_value
# learner: qmix 网络
# pixels：被攻击智能体的数量
def attack_de(img, label, learner, action_values,n_agent, n_action, state_batch,agents_available_actions,
	           pixels=1, maxiter=75, popsize=400, verbose=False):

    target_calss = label
    # print(target_calss)
	# print(agents_available_actions)
    bounds = [(0,n_agent), (0,n_action)] * pixels # len(bounds) = 5
    
    popmul = max(1, popsize//len(bounds))
    # xs, img, learner, action_values, state_batch, agents_available_actions
    predict_fn = lambda xs: predict_classes(
		xs, img, learner, action_values,  state_batch, agents_available_actions) # 要最小化的目标函数
    # x, img, target_calss, learner, action_values, state_batch, verbose=False
    callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, learner, action_values, state_batch, verbose)
    # callback_fn = None

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    for init in inits: # 随机初始化
        for i in range(pixels):
            init[i*2+0] = np.random.random()*n_agent

            init[i*2+1] = np.random.randint(0,n_action,1) 

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

    attack_image = perturb_actions(attack_result.x.astype(int), img.clone())

    return attack_image[0], attack_result.x.astype(int)
