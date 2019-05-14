import os
import json
import math
import torch
import torchnet as tnt

from common import data_utils, model_utils

def eval(net, test_dataloader, logger, args):
    net.eval()

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        args.test_way, args.test_shot, args.test_query, args.test_episodes ))

    meters = { field: tnt.meter.AverageValueMeter() for field in args.trace_field }

    model_utils.evaluate(net, test_dataloader['test'], meters, desc="test")

    for field,meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(args.test_episodes)))
