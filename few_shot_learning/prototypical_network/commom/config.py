import os
import torch
import threading

BASE_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
MODEL_ROOT = os.path.join(BASE_DIR,'model')
LOG_ROOT = os.path.join(BASE_DIR,'logs')

#uses a class instead of using parser
class arguments(object):
    _instance_lock = threading.Lock()
    def __init__(self):
        '''
        ////////:param resplit: replit the data
        :param sequential: use sequential sampler instead of episodic
        :param trainval: validation set is merged to the trainning set if True

        :param weight_decay_every: num of epoches for every weight decay
        :param patient: num of epoches to wait before validation begin

        '''
        #if use cuda
        self.cuda = torch.cuda.is_available()
        self.multigpu = True

        #data args
        #self.resplit = False
        self.dataset = 'omniglot'
        self.split_path = os.path.join(BASE_ROOT, self.dataset, 'splits')
        self.splits = 'vinyals'
        self.dataset_path = os.path.join(BASE_ROOT, self.dataset)
        self.sequential = False
        self.train_way = 60
        self.train_shot = 5
        self.query = 5
        self.test_way = 5
        self.test_shot = self.train_shot_n
        self.test_query = 15
        self.train_episodes = 100
        self.test_episodes = 100
        self.trainval =False

        #model args
        self.model_name = 'protonet_conv'
        self.input_size = [1,28,28]
        self.hidden_size = 64
        self.output_size = 64


        #train args
        #self.batch_size #in few shot learning, episode is used
        self.train_epoches = 10000
        self.opt_method = 'Adam'
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.weight_decay_every = 20
        self.patient = 1000

        #log args
        res_dir = os.path.join(BASE_ROOT, 'results')
        self.models_dir = os.path.join(res_dir,'models')
        self.logs_dir =  os.path.join(res_dir, 'logs')
        self.trace_fields = ['loss','accuracy']
        self.trace_dir = os.path.join(res_dir,'traces')

    # sigleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(arguments, "_instance"):
            with arguments._instance_lock:
                if not hasattr(arguments, "_instance"):
                    arguments._instance = object.__new__(cls)
        return arguments._instance

    def to_dict(self):
        d = {}
        for k,v in vars(self):
            d[k] = v
        return d

args = arguments()