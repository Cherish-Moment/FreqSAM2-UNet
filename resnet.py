import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, type: int = 50, is_pretrained: bool = True):
        super(ResNet, self).__init__()

        original_model = None
        self.channel_list = None

        if type == 50:
            if is_pretrained:
                original_model = models.resnet50(pretrained=True)
                self.channel_list = [256, 512, 1024, 2048]
            else:
                original_model = models.resnet50(pretrained=False)
                self.channel_list = [256, 512, 1024, 2048]
        elif type == 34:
            if is_pretrained:
                original_model = models.resnet34(pretrained=True)
                self.channel_list = [64, 128, 256, 512]
            else:
                original_model = models.resnet34(pretrained=False)
                self.channel_list = [64, 128, 256, 512]

        # 修改 ResNet 的层以适应特定的输出尺寸
        self.stage0 = nn.Sequential(*list(original_model.children())[:4])  # 输入层到第一个卷积层
        self.stage1 = nn.Sequential(original_model.layer1)
        self.stage2 = nn.Sequential(original_model.layer2)
        self.stage3 = nn.Sequential(original_model.layer3)
        self.stage4 = nn.Sequential(original_model.layer4)

        # 自定义的下采样层，以确保每个阶段的输出通道数
        self.downsample1 = nn.Conv2d(256, 144, kernel_size=1, stride=1)  # 确保输出通道为144
        self.downsample2 = nn.Conv2d(512, 288, kernel_size=1, stride=1)  # 确保输出通道为288
        self.downsample3 = nn.Conv2d(1024, 576, kernel_size=1, stride=1)  # 确保输出通道为576
        self.downsample4 = nn.Conv2d(2048, 1152, kernel_size=1, stride=1)  # 确保输出通道为1152

    def forward(self, x):
        x = self.stage0(x)  # 输入的初始卷积层
        x = self.stage1(x)  # ResNet第一层输出
        f1 = self.downsample1(x)  # 将输出通道调整为144
        x = self.stage2(x)  # ResNet第二层输出
        f2 = self.downsample2(x)  # 将输出通道调整为288
        x = self.stage3(x)  # ResNet第三层输出
        f3 = self.downsample3(x)  # 将输出通道调整为576
        x = self.stage4(x)  # ResNet第四层输出
        f4 = self.downsample4(x)  # 将输出通道调整为1152
        return [f1, f2, f3, f4]
