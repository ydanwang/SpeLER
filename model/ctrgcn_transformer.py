import math
import pdb
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from graph.graph import Graph
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange


class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, **kwargs):
        if 'softmax' in kwargs:
            return self.dense(x)

        return self.softmax(self.dense(x))
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, output_dim=32) -> None:
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x):
        return self.projection_head(x)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), x.size(1))

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels) 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.residual = lambda x: 0

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    '''
    NCTV
    '''
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))#N,C,V,1 - N,C,1,V -> N,C,V,V
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

class FrequencyAwareGatedConv2d(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0):
        super(FrequencyAwareGatedConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.AdaptiveAvgPool2d((10, 50))

    def forward(self, x, cross=False):
        x_conv = self.conv(x)
        x_gate = self.gate(x)

        x_gate = torch.sigmoid(x_gate)

        x_gated = x_conv * x_gate

        if cross:
            output = self.pool(x_gated)
            return output
        return x_gated
    
class FrequencyAwareGatedConv1d(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0):
        super(FrequencyAwareGatedConv1d, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=1024), 
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.gate = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=1024), 
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x, cross=False):

        if len(x.shape) == 2:
            # [B, C] -> [B, C, 1]
            x = x.unsqueeze(-1)
        elif len(x.shape) == 4:
            # [B, C, H, W] -> [B, C, 1]
            x = x.mean(dim=(2, 3)).unsqueeze(-1)
            
        if x.shape[1] != self.conv[0].in_channels:
            x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]

        x_conv = self.conv(x)  
        x_gate = self.gate(x)  
        
        x_gated = x_conv * x_gate
        
        if cross:
            output = self.pool(x_gated)  
            output = output.squeeze(-1)   
            return output
            
        return x_gated

class Time_branch(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2], num_layer=4, num_head=8):
        super(Time_branch, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.gcn2 = unit_gcn(out_channels, out_channels*2, A, adaptive=adaptive)
        self.tcn2 = MultiScale_TemporalConv(out_channels*2, out_channels*2, kernel_size=kernel_size, stride=1, dilations=dilations,
                                            residual=False)

        self.gcn3 = unit_gcn(out_channels*4, out_channels*4, A, adaptive=adaptive)
        self.tcn3 = MultiScale_TemporalConv(out_channels*4, out_channels*4, kernel_size=kernel_size, stride=2, dilations=dilations,
                                            residual=False)

        self.gcn4 = unit_gcn(out_channels*4, out_channels*4, A,  adaptive=adaptive)
        self.tcn4 = MultiScale_TemporalConv(out_channels*4, out_channels*4, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)

        self.gcn5 = unit_gcn(out_channels*8, out_channels*8, A, adaptive=adaptive)
        self.tcn5 = MultiScale_TemporalConv(out_channels*8, out_channels*8, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        
        encoder_layer = TransformerEncoderLayer(out_channels*8, num_head, out_channels*8, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)

        self.gate_intra_1 = FrequencyAwareGatedConv2d(out_channels, out_channels*2)
        self.gate_intra_2 = FrequencyAwareGatedConv2d(out_channels*4, out_channels*4)
        encoder_layer = TransformerEncoderLayer(out_channels*8, num_head, out_channels*8, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)
        
        self.channel_t = nn.Sequential(
            nn.Linear(2560, out_channels*8),
            nn.LayerNorm(out_channels*8),
            nn.ReLU(True),
            nn.Linear(out_channels*8, out_channels*8),
        )

        self.channel_s = nn.Sequential(
            nn.Linear(10240, out_channels*8),
            nn.LayerNorm(out_channels*8),
            nn.ReLU(True),
            nn.Linear(out_channels*8, out_channels*8),
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_block_1 = self.relu(self.tcn1(self.gcn1(x)))
        x_block_2 = self.relu(self.tcn2(self.gcn2(x_block_1)))
        x_2_fused = torch.cat((x_block_2, self.gate_intra_1(x_block_1)), dim=1)

        x_block_3 = self.relu(self.tcn3(self.gcn3(x_2_fused)))
        x_block_4 = self.relu(self.tcn4(self.gcn4(x_block_3))) 
        x_4_fused = torch.cat((x_block_4, self.gate_intra_2(x_block_3)), dim=1)

        x_block_5 = self.relu(self.tcn5(self.gcn5(x_4_fused)))

        vs = rearrange(x_block_5, '(B M) C T V -> B (M V) (T C)', M=1)
        vs = self.channel_s(vs)
        vs = self.s_encoder(vs)
        
        vt = rearrange(x_block_1, '(B M) C T V -> B (M V) (T C)', M=1)

        vt = self.channel_t(vt)

        vs = vs.amax(dim=1)
        vt = vt.amax(dim=1)
  
        return vs, vt
        return x_block_5, x_block_1

class Freq_branch(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2], num_layer=4, num_head=8):
        super(Freq_branch, self).__init__()
        self.tcn1 = MultiScale_TemporalConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.tcn2 = MultiScale_TemporalConv(out_channels, out_channels*2, kernel_size=kernel_size, stride=1, dilations=dilations,
                                            residual=False)

        self.tcn3 = MultiScale_TemporalConv(out_channels*4, out_channels*4, kernel_size=kernel_size, stride=2, dilations=dilations,
                                            residual=False)

        self.tcn4 = MultiScale_TemporalConv(out_channels*4, out_channels*4, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)

        self.tcn5 = MultiScale_TemporalConv(out_channels*8, out_channels*8, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        
        encoder_layer = TransformerEncoderLayer(out_channels*8, num_head, out_channels*8, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)

        self.gate_intra_1 = FrequencyAwareGatedConv2d(out_channels, out_channels*2)
        self.gate_intra_2 = FrequencyAwareGatedConv2d(out_channels*4, out_channels*4)


        self.channel_s = nn.Sequential(
            nn.Linear(10240, out_channels*8),  
            nn.LayerNorm(out_channels*8),
            nn.ReLU(True),
            nn.Linear(out_channels*8, out_channels*8),
        )

        self.channel_t = nn.Sequential(
            nn.Linear(2560, out_channels*8),  
            nn.LayerNorm(out_channels*8),
            nn.ReLU(True),
            nn.Linear(out_channels*8, out_channels*8),
        )

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_block_1 = self.relu(self.tcn1(x))
        x_block_2 = self.relu(self.tcn2(x_block_1))
        x_2_fused = torch.cat((x_block_2, self.gate_intra_1(x_block_1)), dim=1)

        x_block_3 = self.relu(self.tcn3(x_2_fused))
        x_block_4 = self.relu(self.tcn4(x_block_3))
        x_4_fused = torch.cat((x_block_4, self.gate_intra_2(x_block_3)), dim=1)

        x_block_5 = self.relu(self.tcn5(x_4_fused))

        vs = rearrange(x_block_5, '(B M) C T V -> B (M V) (T C)', M=1)
        vs = self.channel_s(vs)
        vs = self.s_encoder(vs)
        
        vt = rearrange(x_block_1, '(B M) C T V -> B (M V) (T C)', M=1)

        vt = self.channel_t(vt)

        vs = vs.amax(dim=1)
        vt = vt.amax(dim=1)
  
        return vs, vt
        return x_block_5, x_block_1

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, hid_dim=64, num_layer=4, num_head=8):   
        super(Model, self).__init__()
        
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        A = self.graph.A 

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn_time = nn.BatchNorm1d(in_channels * num_point)
        self.data_bn_freq = nn.BatchNorm1d(in_channels * 2 * num_point)

        base_channel = hid_dim

        self.time = Time_branch(in_channels, base_channel, A, stride=1, adaptive=adaptive)
        self.freq = Freq_branch(in_channels, base_channel, A, stride=1, adaptive=adaptive)

        self.gate_inter_time = FrequencyAwareGatedConv1d(base_channel, base_channel*8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.gate_inter_freq = FrequencyAwareGatedConv1d(base_channel, base_channel*8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        self.cls_time = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            SqueezeChannels(),
        )
        self.cls_freq = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            SqueezeChannels(),
        )


        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_time, 1)
        bn_init(self.data_bn_freq, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.loss = nn.CrossEntropyLoss().cuda()

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def forward(self, x, x_freq, label=None,feat=False):
        if len(x) == 2:
            x, x_freq = x
        if len(x.shape) == 3:
            bs, step, dim = x.size()    
            self.num_point = dim //3

            x_in = x.view(bs, step, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            
            bs_f, step_f, dim_f = x_freq.size()
            self.num_point_f = dim_f //3

            x_freq_in = x_freq.view(bs_f, step_f, self.num_point_f, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
  
        bs, C, step, V, M = x_in.size()
        bs_f, C_f, step_f, V_f, M_f = x_freq_in.size()

        x_in = self.data_bn_time(x_in.view(bs, M * V * C, step))
        x_in = x_in.view(bs, M, V, C, step).permute(0, 1, 3, 4, 2).contiguous().view(bs * M, C, step, V)
        
        x_freq_in = x_freq_in.permute(0, 4, 3, 1, 2).contiguous().view(bs_f, M_f * V_f * C_f, step_f)
        x_freq_in = self.data_bn_freq(x_freq_in)
        x_freq_in = x_freq_in.view(bs_f, M_f, V_f, C_f, step_f).permute(0, 1, 3, 4, 2).contiguous().view(bs_f * M_f, C_f, step_f, V_f)

        time_feat, time_shallow = self.time(x_in)
        freq_feat, freq_shallow = self.freq(x_freq_in)


        kxa1 = self.gate_inter_time(time_shallow, cross=True)
        feat_time_final = torch.cat((freq_feat, kxa1), dim=1)    

        kxa2 = self.gate_inter_freq(freq_shallow, cross=True)
        feat_freq_final = torch.cat((time_feat, kxa2), dim=1)    

        if label!=None:
            rep_joint = self._generate_mask(x.clone(), bs, M, label)
            rep_joint = rep_joint.detach()
        
        if label!=None:
            return self.fc(x), rep_joint
        if feat:
            return self.fc(x), x

        return {
            'feat_time': feat_time_final,
            'feat_feq': feat_freq_final,
            }