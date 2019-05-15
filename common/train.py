import os
import torchnet as tnt
import torch
from functools import partial
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from base.engine import Engine
from common import model_utils, log_utils

def train(network, dataloader, logger, args):
    logger.save_opt(args)

    if args.trainval:
        train_loader = dataloader['trainval']
        val_loader = None
    else:
        train_loader = dataloader['train']
        val_loader = dataloader['val']

    engine = Engine()

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in args.trace_fields}}

    if val_loader is not None:
        meters['val'] = {field: tnt.meter.AverageValueMeter() for field in args.trace_fields}

    def on_start(state):
        logger.reset()
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], args.weight_decay_every, gamma=0.5)

    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])

    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        logger.save_trace(meter_vals, 'state=train')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(args.models_dir, 'best_model.pt'))
                if args.cuda:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > args.patient:
                    print("==> patience {:d} exceeded".format(args.patience))
                    state['stop'] = True
        else:
            state['model'].cpu()
            logger.save_model(state['model'], args.model_filename)
            if args.cuda:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    engine.train(
        model=network,
        loader=train_loader,
        optim_method=args.optim_method,
        optim_config=args.optim_config,
        max_epoch=args.train_epoches
    )
















