import torch
import numpy as np
import torch.optim as optim

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

    print('Accuracy of the network on the 100 states: %d %%' % (
            100 * correct / total))

BATCH_SIZE = 64
# agent = 'InternalAgent'
agent = 'FlawedAgent'
trainloader = IterableStatesCollection(AGENT_CLASSES,
                                        num_players=3,
                                        agent_id=agent,
                                        batch_size=BATCH_SIZE)

testloader = IterableStatesCollection(AGENT_CLASSES,
                                        num_players=3,
                                        agent_id=agent,
                                        batch_size=BATCH_SIZE,
                                      max_iter=100)

criterion = torch.nn.CrossEntropyLoss()
idx = 0
for net in gen_model(959, 50, [4, 1], [128, 1024]):
    # train model
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
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
            if i % 10 == 9:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            if i % 199 == 198:
                eval_acc(net, testloader)
    path = f'./r2{idx}_net.pth'
    idx += 1
    torch.save(net.state_dict(), path)
    print('Finished Training')

