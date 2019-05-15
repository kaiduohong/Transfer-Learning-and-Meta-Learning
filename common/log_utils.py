import torch
import time
import os
import sys
import json



class LogManager(object):
    def __init__(self,args):
        self.logs_dir = args.logs_dir
        self.models_dir = args.models_dir
        self.trace_file = os.path.join(args.trace_dir, '{}-{}-trace.json'
                                       .format(self.cur_time(),args.trace_filename))

        log_filename = os.path.join(self.logs_dir, 'eval-{}'.format(self.cur_time()))
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(log_filename + '_log.txt')
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    def reset(self):
        if(os.path.exists(self.logs_dir)):
            os.removedirs(self.logs_dir)
        os.mkdir(self.logs_dir)

    def save_model(self,model,model_name):
        model_file = os.path.join(self.models_dir,'{}-{}.pth'
                                  .format(self.cur_time()), model_name)
        torch.save(model.state_dict(), model_file)

    def save_trace(self, meter_vals):
        with open(self.trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

    def cur_time(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_opt(self, opt):
        if not isinstance(opt,dict):
            opt = opt.to_dict()
        with open(os.path.join(self.logs_dir, '{}-opt.json'.format(self.cur_time())), 'w') as f:
            json.dump(opt, f)
            f.write('\n')



def extract_meter_values(meters):
    ret = { }

    for split in meters.keys():
        ret[split] = { }
        for field,meter in meters[split].items():
            ret[split][field] = meter.value()[0]

    return ret

def render_meter_values(meter_values):
    field_info = []
    for split in meter_values.keys():
        for field,val in meter_values[split].items():
            field_info.append("{:s} {:s} = {:0.6f}".format(split, field, val))

    return ', '.join(field_info)

#have not been used
import logging, logging.handlers
class LogMgr(object):
    def __init__(self, logpath, markpath):
        self.LOG = logging.getLogger('log')
        loghdlr1 = logging.handlers.RotatingFileHandler(logpath, "a", 0, 1)
        fmt1 = logging.Formatter("%(asctime)s %(threadName)-10s %(message)s", "%Y-%m-%d %H:%M:%S")
        loghdlr1.setFormatter(fmt1)
        self.LOG.addHandler(loghdlr1)
        self.LOG.setLevel(logging.INFO)

        self.MARK = logging.getLogger('mark')
        loghdlr2 = logging.handlers.RotatingFileHandler(markpath, "a", 0, 1)
        fmt2 = logging.Formatter("%(message)s")
        loghdlr2.setFormatter(fmt2)
        self.MARK.addHandler(loghdlr2)
        self.MARK.setLevel(logging.INFO)

    def error(self, msg):
        if self.LOG is not None:
            self.LOG.error(msg)

    def info(self, msg):
        if self.LOG is not None:
            self.LOG.info(msg)

    def debug(self, msg):
        if self.LOG is not None:
            self.LOG.debug(msg)

    def mark(self, msg):
        if self.MARK is not None:
            self.MARK.info(msg)
