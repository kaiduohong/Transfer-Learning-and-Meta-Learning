
import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch

from network import prototypical_net, train, test
from common import config, data_utils, model_utils, utils

def main():
    logger = utils.LogManager()
    args = config.arguments()
    net = model_utils.load(args)

    torch.manual_seed(1234)
    if args.cuda:
        torch.cuda.manual_seed(1234)
        net = net.cuda()

    train_dataloader = data_utils.dataloader('train')
    test_dataloader = data_utils.dataloader('test')

    print('begin training')
    train(net, train_dataloader, logger)
    print('end training\n\n')

    print('begin eval')
    eval(net,test_dataloader,logger)
    print('end eval')

if __name__ == '__main__':
    main()