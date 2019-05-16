import os
import json
import math
import torch
import torchnet as tnt

from common import data_utils, model_utils,log_utils


def eval(net, test_dataloader, logger, args):
    net.eval()

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        args.test_way, args.test_shot, args.test_query, args.test_episodes ))

    meters = {'test':{ field: tnt.meter.AverageValueMeter() for field in args.trace_fields }}
    model_utils.evaluate(net, test_dataloader['test'], meters['test'], desc="test")

    meter_vals = log_utils.extract_meter_values(meters)

    logger.save_trace(meter_vals, 'test' + args.trace_filename)

    for field,meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(args.test_episodes)))
