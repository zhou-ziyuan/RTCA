import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QCritic_CNN(nn.Module):
    def __init__(self, args):
        super(QCritic_CNN, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        input_shape = self.state_dim + self.n_agents*self.n_actions

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * (((input_shape-2)//2 - 2)//2), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    def forward(self, actions , state):
        # inputs = self._build_inputs(batch, t=t)
        actions = actions.to("cpu")
        state = state.to("cpu")
        # actions = actions.to(self.device)
        # state = state.to(self.device)
        bs = state.size(0)
        # print(bs)
        # print(actions.size())
        # print(actions)
        # print(state)
        x = th.cat([state, actions], dim=-1)

        if (len(x.size())<3):
            x = x.unsqueeze(1)  # 添加一个通道维度

        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)

        qx = self.fc_layers(x)
        q = qx.view(bs, -1, 1)
        return q



class QCritic_CNN_discrete(nn.Module):
    def __init__(self, args):
        super(QCritic_CNN_discrete, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        input_shape = self.state_dim + self.n_agents

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * (((input_shape-2)//2 - 2)//2), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    def forward(self, actions , state):
        # inputs = self._build_inputs(batch, t=t)
        actions = actions.to("cpu")
        state = state.to("cpu")
        # actions = actions.to(self.device)
        # state = state.to(self.device)
        bs = state.size(0)
        # print(bs)
        # print(actions.size())
        # print(actions)
        # print(state)
        x = th.cat([state, actions], dim=-1)

        if (len(x.size())<3):
            x = x.unsqueeze(1)  # 添加一个通道维度

        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)

        qx = self.fc_layers(x)
        q = qx.view(bs, -1, 1)
        return q