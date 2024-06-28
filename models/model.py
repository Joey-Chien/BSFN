import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.multimodal import CLIPImageQualityAssessment
from transformers import AutoImageProcessor, SwinModel
import math



class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

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
        z = W_y + x

        return z



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


    
class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O



class my_SANet(nn.Module):
    def __init__(self, identity=False):
        super(my_SANet, self).__init__()
        self.model_1 = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
        self.model_2 = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
        self.identity = identity
        self.SANet = SANet(1536)


    def forward(self, x):
        sty = self.model_1(x)
        aes = self.model_2(x)
        sty = sty.last_hidden_state
        aes = aes.last_hidden_state
        sty = sty.view(-1, 1536, 49)
        sty = sty.view(-1, 1536, 7, 7)
        aes = aes.view(-1, 1536, 49)
        aes = aes.view(-1, 1536, 7, 7)
        output = self.SANet(aes, sty)
        if self.identity:
            output += aes
        return F.relu(output)



class BSFN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.clip_iqa = CLIPImageQualityAssessment(prompts=("quality", "brightness", "noisiness", "colorfullness", \
                                                                "sharpness", "contrast", "complexity", "natural", "happy", "scary", \
                                                                "new", "warm", "real", "beautiful", "lonely", "relaxing"))
        self.GenAes = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
        self.StyAes = my_SANet()
        self.NLB_1 = NonLocalBlock(in_channels=1536)
        self.NLB_2 = NonLocalBlock(in_channels=3072)
        # self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        # self.bn = nn.BatchNorm2d(num_features=3072)
        # self.ln = nn.LayerNorm(normalized_shape=(3072, 7, 7))  # LayerNorm
        # self.selfAttention = SelfAttention(in_channels=3072)
        # self.CBAM = CBAM(in_planes=3072)
        # self.SNL = SNLStage(inplanes=3072, planes=1536)
        # self.IN = nn.InstanceNorm2d(3072, affine=True)
        self.SWIN_predictor = nn.Sequential(
            nn.Linear(3072 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 111),
            nn.Sigmoid(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Linear(2048, num_classes),
            # nn.Sigmoid(),
        )


    def forward(self, x, q_score):
        output = self.clip_iqa(x)
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x)
        sty_aes = self.NLB_1(sty_aes)
        gen_aes = gen_aes.last_hidden_state
        gen_aes = gen_aes.view(-1, 1536, 49)
        gen_aes = gen_aes.view(-1, 1536, 7, 7)
        all_aes = torch.cat((sty_aes, gen_aes), 1)

        
        all_aes = self.NLB_2(all_aes)


        fc_input = torch.flatten(all_aes, start_dim=1)
        swin_result = self.SWIN_predictor(fc_input)

        # clip
        quality = output['quality']
        brightness = output['brightness']
        noisiness = output['noisiness']
        colorfullness = output['colorfullness']
        sharpness = output['sharpness']
        contrast = output['contrast']
        complexity = output['complexity']
        natural = output['natural']
        happy = output['happy']
        scary = output['scary']
        new = output['new']
        warm = output['warm']
        real = output['real']
        beautiful = output['beautiful']
        lonely = output['lonely']
        relaxing = output['relaxing']

        now_device = 'cuda'

        if quality.dim() == 0:
            quality = torch.tensor([quality.item()], device=now_device)
            brightness = torch.tensor([brightness.item()], device=now_device)
            noisiness = torch.tensor([noisiness.item()], device=now_device)
            colorfullness = torch.tensor([colorfullness.item()], device=now_device)
            sharpness = torch.tensor([sharpness.item()], device=now_device)
            contrast = torch.tensor([contrast.item()], device=now_device)
            complexity = torch.tensor([complexity.item()], device=now_device)
            natural = torch.tensor([natural.item()], device=now_device)
            happy = torch.tensor([happy.item()], device=now_device)
            scary = torch.tensor([scary.item()], device=now_device)
            new = torch.tensor([new.item()], device=now_device)
            warm = torch.tensor([warm.item()], device=now_device)
            real = torch.tensor([real.item()], device=now_device)
            beautiful = torch.tensor([beautiful.item()], device=now_device)
            lonely = torch.tensor([lonely.item()], device=now_device)
            relaxing = torch.tensor([relaxing.item()], device=now_device)
        combined_scores = torch.stack((quality, brightness, noisiness, colorfullness, \
                                        sharpness, contrast, complexity, natural, happy, scary, \
                                        new, warm, real, beautiful, lonely, relaxing), dim=1)

        # qalign
        q_score = q_score.unsqueeze(1)
        q_score = q_score.to(dtype=torch.float32)

        
        final = torch.cat((q_score, combined_scores, swin_result), 1)

        fc_input = torch.flatten(final, start_dim=1)
        result = self.predictor(fc_input)
        return result



class BSFN_AVA(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.clip_iqa = CLIPImageQualityAssessment(prompts=("quality", "brightness", "noisiness", "colorfullness", \
                                                                "sharpness", "contrast", "complexity", "natural", "happy", "scary", \
                                                                "new", "warm", "real", "beautiful", "lonely", "relaxing"))
        self.GenAes = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
        self.StyAes = my_SANet()
        self.NLB_1 = NonLocalBlock(in_channels=1536)
        self.NLB_2 = NonLocalBlock(in_channels=3072)
        # self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        # self.bn = nn.BatchNorm2d(num_features=3072)
        # self.ln = nn.LayerNorm(normalized_shape=(3072, 7, 7))  # LayerNorm
        # self.selfAttention = SelfAttention(in_channels=3072)
        # self.CBAM = CBAM(in_planes=3072)
        # self.SNL = SNLStage(inplanes=3072, planes=1536)
        # self.IN = nn.InstanceNorm2d(3072, affine=True)
        self.SWIN_predictor = nn.Sequential(
            nn.Linear(3072 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 110),
            nn.Sigmoid(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Linear(2048, num_classes),
            # nn.Sigmoid(),
        )


    def forward(self, x, q_score, A_q_score):
        output = self.clip_iqa(x)
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x)
        sty_aes = self.NLB_1(sty_aes)
        gen_aes = gen_aes.last_hidden_state
        gen_aes = gen_aes.view(-1, 1536, 49)
        gen_aes = gen_aes.view(-1, 1536, 7, 7)
        all_aes = torch.cat((sty_aes, gen_aes), 1)

        
        all_aes = self.NLB_2(all_aes)


        fc_input = torch.flatten(all_aes, start_dim=1)
        swin_result = self.SWIN_predictor(fc_input)

        # clip
        quality = output['quality']
        brightness = output['brightness']
        noisiness = output['noisiness']
        colorfullness = output['colorfullness']
        sharpness = output['sharpness']
        contrast = output['contrast']
        complexity = output['complexity']
        natural = output['natural']
        happy = output['happy']
        scary = output['scary']
        new = output['new']
        warm = output['warm']
        real = output['real']
        beautiful = output['beautiful']
        lonely = output['lonely']
        relaxing = output['relaxing']

        now_device = 'cuda'

        if quality.dim() == 0:
            quality = torch.tensor([quality.item()], device=now_device)
            brightness = torch.tensor([brightness.item()], device=now_device)
            noisiness = torch.tensor([noisiness.item()], device=now_device)
            colorfullness = torch.tensor([colorfullness.item()], device=now_device)
            sharpness = torch.tensor([sharpness.item()], device=now_device)
            contrast = torch.tensor([contrast.item()], device=now_device)
            complexity = torch.tensor([complexity.item()], device=now_device)
            natural = torch.tensor([natural.item()], device=now_device)
            happy = torch.tensor([happy.item()], device=now_device)
            scary = torch.tensor([scary.item()], device=now_device)
            new = torch.tensor([new.item()], device=now_device)
            warm = torch.tensor([warm.item()], device=now_device)
            real = torch.tensor([real.item()], device=now_device)
            beautiful = torch.tensor([beautiful.item()], device=now_device)
            lonely = torch.tensor([lonely.item()], device=now_device)
            relaxing = torch.tensor([relaxing.item()], device=now_device)
        combined_scores = torch.stack((quality, brightness, noisiness, colorfullness, \
                                        sharpness, contrast, complexity, natural, happy, scary, \
                                        new, warm, real, beautiful, lonely, relaxing), dim=1)

        # qalign
        q_score = q_score.unsqueeze(1)
        q_score = q_score.to(dtype=torch.float32)
        A_q_score = A_q_score.unsqueeze(1)
        A_q_score = A_q_score.to(dtype=torch.float32)

        
        final = torch.cat((q_score, A_q_score, combined_scores, swin_result), 1)

        fc_input = torch.flatten(final, start_dim=1)
        result = self.predictor(fc_input)
        return result