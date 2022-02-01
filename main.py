import gym
import pyglet
import matplotlib.pyplot as plt
import numpy as np
import random
plt.style.use('_mpl-gallery')

from torch import nn
import torch.nn.functional as F

import torch

import statistics


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
#torch.autograd.set_detect_anomaly(True)

# loads envirorment
env = gym.make('CartPole-v1')
number_of_episode = 300
max_time = 5000

resulting_time_ls = []

# set q earning values
alpha = .5
gamma = .99

# init networks
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))

        x = self.predict(x)             # linear output
        return x

net = NeuralNetwork(4,20,2).to(device)

learning_rate = .05

# set optimizers
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


# temp
# define action list
action_ls_ = [0, 1]


for i_episode in range(number_of_episode):
    total_reward = 0

    observation = env.reset()

    action_prob_ls = []
    dis_reward_ls = []
    state_ls = []
    action_ls = []


    for t in range(max_time):   # game loop

        #env.render()  # loads game window not needed

        # get pre state
        state = observation

        # policy
        with torch.no_grad():
            pred = net(torch.FloatTensor(state))
        softmax = nn.Softmax(dim=-1)
        pred_probab = softmax(pred)
        #print(pred,pred_probab)
        action = random.choices(action_ls_,pred_probab.tolist(),k=1)[0]
        action_prob = pred_probab[action]

        # update envirorment
        observation, reward, done, info = env.step(action)

        total_reward += reward

        discounted_reward = total_reward * gamma

        dis_reward_ls.append(discounted_reward)
        action_prob_ls.append(action_prob)
        state_ls.append(state)
        action_ls.append(action)



        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print(i_episode)
            print_counter = 0
            resulting_time_ls.append(format(t + 1))
            break

    mean = statistics.mean(dis_reward_ls)
    #print(dis_reward_ls)
    for i in range(len(dis_reward_ls)):
        try:
            neg = (mean - dis_reward_ls[i])/abs(mean - dis_reward_ls[i])
        except:
            neg = 0

        #print(neg)
        dis_reward_ls[i] = neg * (abs(mean - dis_reward_ls[i]) ** 1)
    #print(dis_reward_ls)



    state_tensor = torch.FloatTensor(state_ls)
    dis_reward_tensor = torch.FloatTensor(dis_reward_ls)
    action_tensor = torch.reshape(torch.LongTensor(action_ls), [len(action_ls),1])

    optimizer.zero_grad()

    pred_tensor = net(state_tensor)
    softmax = nn.Softmax(dim=-1)
    pred_prob = softmax(pred_tensor)

    values = pred_prob.gather(1, action_tensor.view(-1, 1)).view(-1)

    losses = torch.log(values) * dis_reward_tensor
    loss = -losses.mean()


    # update network

    loss.backward()
    optimizer.step()

env.close()




# plot
fig, ax = plt.subplots()
x = []
y = []
sum = 0
max_y = 0

moving_avg = []
moving_avg_width = 20

for i in range(len(resulting_time_ls)):
    x.append(i)
    y.append(int(resulting_time_ls[i]))

    sum += int(resulting_time_ls[i])

    if(int(resulting_time_ls[i]) > max_y):
        max_y = int(resulting_time_ls[i])

    if(i > moving_avg_width):
        moving_avg_sum = 0
        for n in range(moving_avg_width):
            moving_avg_sum += int(resulting_time_ls[i - n])
        moving_avg.append(moving_avg_sum/moving_avg_width)
    else:
        moving_avg.append(0)


avg = sum/len(resulting_time_ls)


ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
ax.plot(x, moving_avg, linewidth=2.0, color = 'g')
ax.set(xlim=(0, number_of_episode), xticks=np.arange(1, number_of_episode), ylim=(0, (max_y + 5)), yticks=np.arange(1, (max_y + 5)))


#print(resulting_time_ls)
print("avg: ",avg)

plt.show()
