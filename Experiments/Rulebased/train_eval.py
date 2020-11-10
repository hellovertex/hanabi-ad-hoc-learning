import torch
import numpy as np
import torch.optim as optim
from time import time
from data import IterableStatesCollection, AGENT_CLASSES
from model import gen_model


def eval_acc(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print('Accuracy of the network on 100 states: %d %%' % (
            100 * correct / total))

BATCH_SIZE = 8
# agent = 'InternalAgent'
agent = 'FlawedAgent'
trainloader = IterableStatesCollection(AGENT_CLASSES,
                                        num_players=3,
                                        agent_id=agent,
                                        batch_size=BATCH_SIZE,
                                        max_iter=1000)

testloader = IterableStatesCollection(AGENT_CLASSES,
                                        num_players=3,
                                        agent_id=agent,
                                        batch_size=BATCH_SIZE,
                                        max_iter=50)

criterion = torch.nn.CrossEntropyLoss()
idx = 0
# t_collect = 0
# t_train = 0
for net in gen_model(959, 50, [1, 2], [64, 128, 256]):
    # train model
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        t_iter = time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            print(f'{i} iterations take {time() - t_iter} seconds')
            # t_s = time()
            inputs, labels = data
            # t_collect += time() - t_s
            t = time()
            # print(labels)
            # labels = torch.max(labels, 1)[1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # t_train += time() - t
            if i % 1000 == 999:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                # print(f'collect time = {t_collect}')
                # print(f'train time = {t_train}')
                # t_collect = 0
                # t_train = 0
            if i % 4999 == 4998:
                eval_acc(net, testloader)
    path = f'./r2{idx}_net.pth'
    idx += 1
    torch.save(net.state_dict(), path)
    print('Finished Training')

