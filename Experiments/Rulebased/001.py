import torch
import torch.optim as optim
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from data import IterableStatesCollection, AGENT_CLASSES
from model import get_model

# tune overview: https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
# tune random API: https://docs.ray.io/en/master/tune/api_docs/search_space.html#tune-sample-docs

default_search_space = {
    'num_hidden_layers': [1, 2],
    # 'layer_sizes': [i for i in range(64, 1024)],
    'layer_sizes': [64, 96, 128, 196, 256, 376, 448, 512],
    # 'halve_layer_sizes': [True, False],
    'learning_rate': [1e-4, 1e-3],
    'batch_size': [4, 8, 16, 32]
}


def train(config):
    accuracy = config['lr'] * config['num_hidden_layers']
    tune.report(acc=accuracy)


def eval(net, testloader, criterion):
    correct = 0
    total = 0
    eval_steps = 0
    eval_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            eval_loss += criterion(outputs, y).numpy()
            eval_steps += 1
            total += y.size(0)
            correct += (predicted == y).sum().item()
    # acc = 100 * correct / total
    # print('Accuracy of the network on 100 states: %d %%' % (acc))

    # return (loss, accuracy)
    return (eval_loss / eval_steps, correct / total)


def train_eval(config,
               max_train_epochs=20,
               eval_interval=200,
               max_dataset_size=1e4,  # do not change this
               max_eval_iter=50):
    agent = config['agent']
    lr = config['lr']
    num_hidden_layers = config['num_hidden_layers']
    layer_size = config['layer_size']
    batch_size = config['batch_size']
    num_players = config['num_players']
    trainloader = IterableStatesCollection(AGENT_CLASSES,
                                           num_players=num_players,
                                           agent_id=agent,
                                           batch_size=batch_size,
                                           max_iter=max_dataset_size)

    testloader = IterableStatesCollection(AGENT_CLASSES,
                                          num_players=num_players,
                                          agent_id=agent,
                                          batch_size=batch_size,
                                          max_iter=max_eval_iter)

    model = get_model(observation_size=959,
                      num_actions=50,
                      num_hidden_layers=num_hidden_layers,
                      layer_size=layer_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(max_train_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % eval_interval == 0:
                eval_loss, eval_acc = eval(model, testloader, criterion)
                tune.report(loss=eval_loss, acc=eval_acc)




def main():
    config = {'agent': 'FlawedAgent',
              'lr': tune.loguniform(1e-4, 1e-1),
              'num_hidden_layers': tune.grid_search([1, 2]),
              'layer_size': tune.grid_search([64, 96, 128, 196, 256, 376, 448, 512]),
              'batch_size': tune.choice([4, 8, 16, 32]),
              'num_players': 3}
    pass


search_space = {'lr': tune.loguniform(1e-4, 1e-1),
                'num_hidden_layers': tune.grid_search([1, 2]),
                'layer_sizes': tune.grid_search([64, 96, 128, 196, 256, 376, 448, 512]),
                }
analysis = tune.run(train, config=search_space, num_samples=10, scheduler=ASHAScheduler())
dfs = analysis.trial_dataframes
print(len(dfs))
# [d.acc.plot() for d in dfs.values()]
