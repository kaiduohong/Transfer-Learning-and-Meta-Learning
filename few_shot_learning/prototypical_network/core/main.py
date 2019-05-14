
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append('../../../')
print(os.path.dirname('../../../'))
sys.path.append(os.path.dirname('..'))


import torch


from Transfer_Learning_and_Meta_Learning.common import data_utils, model_utils, utils, train, eval
from few_shot_learning.common import config


def main():
    args = config.arguments()
    logger = utils.LogManager(args)
    net = model_utils.load(args)
    args.optim_config = {'lr': args.learning_rate,
     'weight_decay': args.weight_decay}
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devide

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        net = net.cuda()

    train_dataloader = data_utils.dataloader('train')
    test_dataloader = data_utils.dataloader('test')

    print('begin training')
    train.train(net, train_dataloader, logger, args)
    print('end training\n\n')

    print('begin eval')
    eval.eval(net, test_dataloader, logger, args)
    print('end eval')

if __name__ == '__main__':
    main()