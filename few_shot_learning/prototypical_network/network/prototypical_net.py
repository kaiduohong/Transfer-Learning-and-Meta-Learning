import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from few_shot_learning.common.config import args
from common.utils import euclidean_dist
from base.factory import register_model


@register_model('protonet_conv')
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()

        input_size = args.input_size
        hidden_size = args.hidden_size
        output_size = args.output_size

        encoder = nn.Sequential(
            self.conv_block(input_size[0], hidden_size),
            self.conv_block(hidden_size, hidden_size),
            self.conv_block(hidden_size, hidden_size),
            self.conv_block(hidden_size, output_size),
        )

    def forward(self, input):
        x = self.encoder.forward(input)
        return x.view(x.size(0), -1)


    def loss(self, sample):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
        output = self.forward(x)
        output_size = output.size(-1)

        C_proto = output[:n_class * n_support].view(n_class, n_support, output_size).mean(1)  # C n_class * z_dim
        query_embed = output[n_class * n_support:]

        dists = euclidean_dist(query_embed, C_proto)  # [n_class*n_spport, n_class]
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)  # [n_class, n_spport, n_class]
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), #3*3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )



