
import os
import sys
sys.path.append('.')
sys.path.append(os.path.dirname('..'))
sys.path.append('../../../../')
sys.path.append('../../../')


import torch


from common import data_utils, model_utils,log_utils, utils, train, eval
from few_shot_learning.common import config
from few_shot_learning.prototypical_network.network import prototypical_net


def main():
    args = config.arguments()
    logger = log_utils.LogManager(args)
    net = model_utils.load(args.model_name)
    if args.cuda:
        torch.cuda.set_device(args.cuda_devide)

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        net = net.cuda()

    args.state = 'train'
    train_dataloader = data_utils.get_dataloader(args)
    args.state = 'test'
    test_dataloader = data_utils.get_dataloader(args)

    print('begin training')
    train.train(net, train_dataloader, logger, args)
    print('end training\n\n')

    print('begin eval')
    eval.eval(net, test_dataloader, logger, args)
    print('end eval')

if __name__ == '__main__':
    main()
