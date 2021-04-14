import torch
from torch import nn
from torch.nn import functional as F


class NLB(nn.Module):
    '''
    Non-Local block
    '''

    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=False):
        super(NLB, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.name = 'NLB'

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.fc1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, EB=False, bn_layer=False):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        x_residual = x

        # proposed method
        if EB:
            x = EB(x, bn_layer)
            x = nn.ReLU()(x)

        else:
            x = self.bn(x)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x_residual

        return z


class Down_Conv(nn.Module):
    def __init__(self, in_planes, planes):
        super(Down_Conv, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.name = 'Down_Conv'

    def forward(self, x, EB=False):
        return self.bn(self.conv(x))


class Self_Attention_res56(nn.Module):
    def __init__(self, _Encoder, _NLB, _Classifier, _Down_Conv, _num_sa_blocks, _Global_attribute=False):
        super(Self_Attention_res56, self).__init__()

        self.Encoder = _Encoder(num_sa_block=_num_sa_blocks)
        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_num_sa_blocks, _NLB, _Down_Conv)

        if _Global_attribute:
            if _num_sa_blocks > 18:
                self.bn3 = nn.BatchNorm2d(16)
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB_2d()

            elif _num_sa_blocks > 9:
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB_2d()

            else:
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB_2d()

        else:
            self.EB = False

    def make_sa_layer(self, num_sa_blocks, NLB, Down_Conv):
        NLB_list_tmp = []
        if num_sa_blocks < 9:
            for i in range(num_sa_blocks):
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 18 and num_sa_blocks >= 9:
            for i in range(num_sa_blocks - 9):
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 27 and num_sa_blocks >= 18:
            for i in range(num_sa_blocks - 18):
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        else:
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(3, 16))
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = self.Encoder(x)
        for i in range(len(self.NLB_list)):
            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn3)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn2)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn1)
                    else:
                        latent = self.NLB_list[i](latent)
            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out


class Self_Attention_full(nn.Module):
    def __init__(self, _NLB, _Classifier, _Down_Conv, _Global_attribute=False):
        super(Self_Attention_full, self).__init__()

        self.fc1 = nn.Conv2d(3, 16, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_NLB, _Down_Conv)

        if _Global_attribute:

            self.bn3 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn1 = nn.BatchNorm2d(64)
            self.EB = EB_2d()

        else:
            self.EB = False

    def make_sa_layer(self, NLB, Down_Conv):
        NLB_list_tmp = []

        for i in range(9):
            NLB_list_tmp.append(NLB(16))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(16, 32))
            NLB_list_tmp.append(NLB(32))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(32, 64))
            NLB_list_tmp.append(NLB(64))

        return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = F.relu(self.bn0(self.fc1(x)))

        for i in range(len(self.NLB_list)):
            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn3)
                    else:
                        latent = self.NLB_list[i](latent, self.EB)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn2)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn1)
                    else:
                        latent = self.NLB_list[i](latent)
            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out
