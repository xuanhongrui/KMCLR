import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from Mul_Par import args
import numpy as np


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class MetaWeightNet(nn.Module):
    def __init__(self, beh_num):
        super(MetaWeightNet, self).__init__()

        self.beh_num = beh_num

        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.LeakyReLU(negative_slope=args.slope)  
        self.prelu = torch.nn.PReLU()
        self.relu = torch.nn.ReLU()
        self.tanhshrink = torch.nn.Tanhshrink()
        self.dropout7 = torch.nn.Dropout(args.drop_rate)
        self.batch_norm = torch.nn.BatchNorm1d(1)

        initializer = nn.init.xavier_uniform_

        self.SSL_layer1 = nn.Linear(args.hidden_dim*3, int((args.hidden_dim*3)/2))
        self.SSL_layer2 = nn.Linear(int((args.hidden_dim*3)/2), 1)
        self.SSL_layer3 = nn.Linear(args.hidden_dim*2, 1)

        self.RS_layer1 = nn.Linear(args.hidden_dim*3, int((args.hidden_dim*3)/2))
        self.RS_layer2 = nn.Linear(int((args.hidden_dim*3)/2), 1)
        self.RS_layer3 = nn.Linear(args.hidden_dim, 1)

        self.beh_embedding = nn.Parameter(initializer(torch.empty([beh_num, args.hidden_dim]))).cuda()
 

    def forward(self, infoNCELoss_list, behavior_loss_multi_list, user_step_index, user_index_list, user_embeds, user_embed):  
        
        infoNCELoss_list_weights = [None]*self.beh_num
        behavior_loss_multi_list_weights = [None]*self.beh_num
        for i in range(self.beh_num):

            user_index_list[i] = user_index_list[i].long()

            SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)
            SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1)
            SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))

            infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))  
            infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())

            infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
            infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])
            SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
            SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()

            SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
            infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2

            RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1)
            RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)
            RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]])
            behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
            behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze())
            behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(1))
            behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
            RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
            RS_weight3 = self.batch_norm(RS_weight3).squeeze()
            RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
            behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3


        return infoNCELoss_list_weights, behavior_loss_multi_list_weights



