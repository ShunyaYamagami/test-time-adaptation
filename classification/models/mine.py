# https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, input):
        output = self.model(input)
        return output


class MineTrainer():
    def __init__(self, mine):
        super().__init__()
        self.mine = mine
        # self.mine_optim = optimizer
        # self.mine_optim = optim.Adam(self.mine.parameters(), lr=1e-3)
        self.ma_rate=0.01
        self.batch_size = 100
        self.ma_et = 1.

    # OK MINE.ipynb そのまま
    def mutual_information(self, joint, marginal):
        t = self.mine(joint)
        et = torch.exp(self.mine(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et
    
    # OK
    def sample_batch(self, data, sample_mode='joint'):
        if sample_mode == 'joint':
            index = np.random.choice(range(data.shape[0]), size=self.batch_size, replace=False)
            batch = data[index]
        else:
            joint_index = np.random.choice(range(data.shape[0]), size=self.batch_size, replace=False)
            marginal_index = np.random.choice(range(data.shape[0]), size=self.batch_size, replace=False)
            batch = torch.cat([data[joint_index][:,0].reshape(-1,1),
                                data[marginal_index][:,1].reshape(-1,1)],
                                dim=1)
        return batch
    
    # OK
    def ma(self, a, window_size=100):
        # return [torch.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]
        
    def get_loss(self, data):
        # batch is a tuple of (joint, marginal)
        joint = self.sample_batch(data, 'joint').float()
        marginal = self.sample_batch(data, 'marginal').float()
        mi_lb, t, et = self.mutual_information(joint, marginal)
        self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * torch.mean(et)
        self.ma_et = self.ma_et.detach()
        
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1 / self.ma_et.mean()) * torch.mean(et))
        return loss, mi_lb


    def train(self, data, iter_num=int(5e+3)):
        optimizer = torch.optim.Adam(self.mine.parameters(), lr=1e-3)

        result = []
        for i in range(iter_num):
            optimizer.zero_grad()
            loss, mi_lb = self.get_loss(data)
            loss.backward()
            optimizer.step()

        #     result.append(mi_lb.detach().cpu().numpy())
        # return result

