from tqdm import tqdm
from base import get_model

def load(model_name,*args):
    return get_model(model_name, *args)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()
    for field, meter in meters.items(): 
         meter.reset()

    #设置进度条
    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
