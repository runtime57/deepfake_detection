import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, 
                 symm_chomp = False, no_padding = False, relu_type = 'prelu', dwpw=False):
        super(TemporalBlock, self).__init__()
        
        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2*padding-4
            padding = 1 # hack-ish thing so that we can use 3 layers

        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding,symm_chomp)  if not self.no_padding else None
        self.relu1 = nn.PReLU(num_parameters=n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding,symm_chomp) if not self.no_padding else None
        self.relu2 = nn.PReLU(num_parameters=n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
    
        if self.no_padding:
            self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                        self.conv2, self.batchnorm2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                        self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size,True)
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout=0.2, relu_type='prelu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = 3
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                     padding=(self.ksize-1) * dilation_size, dropout=dropout, symm_chomp = True,
                                     no_padding = False, relu_type=relu_type, dwpw=dwpw) )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def reshape_tensor(x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch * s_time, n_channels, sx, sy)
    

class MSTCN(nn.Module):
    """
    Multi-Scale TCN classifier head
    """
    def __init__(self, input_size, num_channels=[256, 256, 256, 256], num_classes=2):
        super(MSTCN, self).__init__()

        self.ksize = 3
        self.num_kernels = 1

        self.ms_tcn = TemporalConvNet(
            input_size, num_channels
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        """
        Args
        x torch.Tensor: input features shaped BxDxT

        Returns
        logits torch.Tensor: probabilities shaped Bx2
        """
        out = self.ms_tcn(x)
        out = out.mean(dim=-1)
        return self.tcn_output(out)
