import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
gradients = {}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
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

    
class FrequencyAwareGatedConv1d(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0):
        super(FrequencyAwareGatedConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x_conv = self.conv(x)
        x_gate = self.gate(x)

        x_gate = torch.sigmoid(x_gate)

        x_gated = x_conv * x_gate

        return x_gated
    
class FlexibleAttentionFusion(nn.Module):
    def __init__(self, hid_dim):
        super(FlexibleAttentionFusion, self).__init__()
        self.attn = nn.Linear(hid_dim*2, 1)
        self.transform = nn.Conv1d(hid_dim, hid_dim*2, kernel_size=1)

    def forward(self, x1, x2):
        x1_transformed = self.transform(x1)

        attn_weights = self.attn(x2.transpose(1, 2)) 
        attn_weights = F.softmax(attn_weights, dim=1) 

        x1_weighted = attn_weights.transpose(1, 2) * x1_transformed

        fused_output = x1_weighted + x2  
        return fused_output, attn_weights



class FCN_expert_residual_gt_pymaid(nn.Module):

    def __init__(self, num_classes, input_size=1, hid_dim=128, class_weights=1, device='cuda', configs=None):
        super(FCN_expert_residual_gt_pymaid, self).__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.hid_dim = hid_dim
        self.dropout = 0.25

        #  time
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hid_dim, kernel_size=configs.kernel_size, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim, out_channels=hid_dim*2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hid_dim*2),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )
       
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*2, out_channels=hid_dim*4, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hid_dim*4),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*4, out_channels=hid_dim*2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hid_dim*2),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*2, out_channels=hid_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )

        #  feq
        self.conv_block1_feq = nn.Sequential(
            nn.Conv1d(in_channels=input_size*2, out_channels=hid_dim, kernel_size=configs.kernel_size, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        self.conv_block2_feq = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim, out_channels=hid_dim*2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hid_dim*2),
            nn.ReLU(),
        )
        self.conv_block3_feq = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*2, out_channels=hid_dim*4, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hid_dim*4),
            nn.ReLU(),

        )
        self.conv_block4_feq = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*4, out_channels=hid_dim*2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hid_dim*2),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )
        self.conv_block5_feq = nn.Sequential(
            nn.Conv1d(in_channels=hid_dim*2, out_channels=hid_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
        )
        self.cls_feq = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )

        self.gate_inter_time = FrequencyAwareGatedConv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
        self.gate_inter_feq = FrequencyAwareGatedConv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
        
        self.gate_intra_time_1 = FlexibleAttentionFusion(hid_dim)
        self.gate_intra_time_2 = FlexibleAttentionFusion_2(hid_dim)
        self.gate_intra_feq_1 = FlexibleAttentionFusion(hid_dim)
        self.gate_intra_feq_2 = FlexibleAttentionFusion_2(hid_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
                
    def forward(self, x, x_feq, **kwargs):
        #### time layer 1
        out_time_1 = self.conv_block1(x) 

        #### feq layer 1
        out_freq_1 = self.conv_block1_feq(x_feq) 

        #### time layer 2
        out_time_2 = self.conv_block2(out_time_1)  

        #### feq layer 2
        out_freq_2 = self.conv_block2_feq(out_freq_1)

        #### time layer 3
        fused_output_time_1, attn_weights_time_1 = self.gate_intra_time_1(out_time_1, out_time_2)
        out_time_3 = self.conv_block3(fused_output_time_1)

        #### feq layer 3
        fused_output_freq_1, attn_weights_freq_1 = self.gate_intra_feq_1(out_freq_1, out_freq_2)

        out_freq_3 = self.conv_block3_feq(fused_output_freq_1)


        #### time layer 4
        
        out_time_4 = self.conv_block4(out_time_3)

        #### feq layer 4

        out_freq_4 = self.conv_block4_feq(out_freq_3)

        #### time layer 5
        fused_output_time_2, attn_weights_time_2 = self.gate_intra_time_2(out_time_3, out_time_4)

        out_time_5 = self.conv_block5(fused_output_time_2)
        feat_time_5 = self.cls(torch.cat((out_time_5, self.gate_inter_time(out_freq_1)), dim=2))

        #### feq layer 5
        fused_output_freq_2, attn_weights_freq_2 = self.gate_intra_feq_2(out_freq_3, out_freq_4)

        out_freq_5 = self.conv_block5_feq(fused_output_freq_2)
        feat_freq_5 = self.cls(torch.cat((out_freq_5, self.gate_inter_feq(out_time_1)), dim=2))


        return {
            'feat_time': feat_time_5,
            'feat_feq': feat_freq_5,
            }
