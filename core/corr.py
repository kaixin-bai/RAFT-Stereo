import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    """
    这段代码定义了一个名为PytorchAlternateCorrBlock1D的类，该类用于计算两个输入特征图之间的1D卷积相关性。它的主要目的是构建一个金字塔结构，将
    两个输入特征图通过空间坐标的变换进行逐层卷积相关计算，然后将计算结果沿着深度维度拼接在一起返回。
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        初始化实例。其中，fmap1和fmap2分别是两个输入特征图，num_levels表示金字塔的层数，默认为4，radius表示每一层金字塔对应的空间坐标变换范
        围，默认为4。
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        """
        todo: 修改这个函数，使得corr的计算更高效
        计算输入特征图fmap1和fmap2之间的卷积相关性。coords是一个包含空间坐标的张量，表示需要在fmap2上采样的位置。首先，将coords的坐标映射到
        范围[-1, 1]内，然后使用F.grid_sample函数对fmap2进行采样，得到采样后的特征图fmapw_mini。接下来，对fmapw_mini和fmap1进行逐通道的
        点乘并求和，得到卷积相关性结果。最后，除以特征图通道数D的平方根进行归一化。
        """
        B, D, H, W = fmap2.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)  # [1,136,240,9,1]
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)  # [1,136,240,9,2]
        output_corr = []
        for grid_slice in grid.unbind(3):  # grid_slice:[1,136,240,2]
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)  # fmapw_mini:[1,256,136,240]
            corr = torch.sum(fmapw_mini * fmap1, dim=1)  # corr:[1,136,240]
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)  # [1,136,240,9]

        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        """
        用于计算两个输入特征图之间的1D卷积相关性金字塔。coords是一个包含空间坐标的张量。首先，对coords进行一系列的维度变换和处理，得到用于计算
        1D卷积相关性的坐标coords_lvl。然后，通过循环计算金字塔的每一层卷积相关性，将结果存储在out_pyramid列表中。每一层金字塔都会对fmap2进
        行平均池化，以便下一层计算。最后，将所有层的卷积相关性结果沿深度维度进行拼接，并按照指定的维度顺序进行变换，最终返回1D卷积相关性金字塔结果。
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # shape from [1,2,136,240] to [1,136,240,2]
        batch, h1, w1, _ = coords.shape  # _:2  h1:136  batch:1  w1:240
        fmap1 = self.fmap1  # [1,256,136,240]
        fmap2 = self.fmap2
        out_pyramid = []
        for i in range(self.num_levels):
            """
            下面注释为当i=0时
            """
            dx = torch.zeros(1)  # dx:tensor([0.])
            dy = torch.linspace(-r, r, 2*r+1)  # dy:tensor([-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.])
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)  # delta是从[-4,0]到[4,0]  # [9,1,2]
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()  # 从[0,0],[1,0]...[239,0]开始到[0,1],[1,1]...[239,1]再到[0,135],[1,135]...[239,135]  # [1,136,240,1,2]
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i  # [1,136,240,1,2]
            coords_lvl = centroid_lvl + delta.view(-1, 2)  # [1,136,240,9,2]
            corr = self.corr(fmap1, fmap2, coords_lvl)  # [1,136,240,9]
            # """
            # corr可视化
            # """
            # if i == 0:
            #     import matplotlib.pyplot as plt
            #     corr_numpy = corr.cpu().numpy()
            #     for j in range(9):
            #         plt.subplot(3, 3, j + 1)  # 创建一个3x3的子图网格，并定位到第i+1个图
            #         plt.imshow(corr_numpy[0, :, :, j], cmap='jet')  # 画出第i个通道的图像
            #         plt.title(f'Channel {j + 1}')  # 设置子图的标题
            #         plt.axis('off')  # 隐藏坐标轴
            #     plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域。
            #     plt.show()
            # =====================================================
            # 对fmap2进行下采样
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])  # 在循环中从[1,256,136,240] ==> [1,256,136,120] ==> [1,256,136,60] ==> [1,256,136,30]
            out_pyramid.append(corr)  # out_pyramid:{list4},每个都是tensor[1,136,240,9]
        out = torch.cat(out_pyramid, dim=-1)  # out:[1,136,240,36], 36是4*9
        return out.permute(0, 3, 1, 2).contiguous().float()


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
