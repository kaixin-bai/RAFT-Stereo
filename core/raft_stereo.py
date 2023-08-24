import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims  # [128,128,128]
        # cnet是context encoder
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm,
                                      downsample=args.n_downsample)  # n_downsample=2； context_norm=‘batch’； context_dims：[128,128,128];hidden_dims:[128,128,128]
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # 列表推导式 [nn.Conv2d(...)... for i in range(self.args.n_gru_layers)] 是用于创建卷积层的。这些卷积层的数量与n_gru_layers一致。
        # 输入通道数是context_dims[i]，对应于每个GRU层的上下文维度。
        # 输出通道数是args.hidden_dims[i] * 3，因为GRU有三个门（更新门、重置门和新的隐藏状态），所以我们需要三倍于隐藏维度的输出通道。
        # 核大小是3x3，padding设置为1或2，具体取决于n_gru_layers的值。
        # z, r, 和 h 或 q 通常代表不同的门或状态。具体来说，z 通常代表更新门的激活，r 代表重置门的激活，而 h 或 q 可能代表新的候选隐藏状态。
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        if args.shared_backbone:  # shared_backbone为False
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            # fnet是feature encoder
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # [1,3,544,960]  将图像从0~255的取值变成了-1~1，这样预处理的原因是：tanh激活函数的输出范围是-1～1，所以将输入也调整到这个范围有助于网络的训练
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        # 使用 autocast 上下文管理器，它可以在混合精度模式下自动调整张量的数据类型，从而提高训练速度和效率。在这里，它被设置为由 self.args.mixed_precision 控制（在此例中为 False，因此不会启用混合精度）。
        # todo: 如果我们要做修改的话，这里下面的cnet和fnet是我们要修改的部分！
        with autocast(enabled=self.args.mixed_precision):  # mixed_precision是True
            if self.args.shared_backbone:  # shared_backbone是False
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True,
                                          num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                # tuple3,其中每个为list2,其中为[1,128,136,240][1,128,68,120][1,128,34,60],128维H/4,H/8,H/16的特征图
                # 这里是list2的原因是在cnet中最后那里有两个结构一样(但是分别实例化)的分支(说明他们的权重不share)，这两个分支是分别在后面计算net_list和inp_list激活用的
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])  # fmap1和fmap2的shape都是[1,256,136,240]
            # cnet_list是tupe3，里面有不同下采样的特征图，在cnet中最后有两个一模一样的分支结构，用来分别做提取，分别做激活.
            net_list = [torch.tanh(x[0]) for x in
                        cnet_list]  # tanh将输入的数值压缩到-1和1之间，list3:[1,128,136,240],[1,128,68,120],[1,128,34,60]
            inp_list = [torch.relu(x[1]) for x in
                        cnet_list]  # relu对负数的处理方式是直接归零，list3:[1,128,136,240],[1,128,68,120],[1,128,34,60]

            # 这段代码主要是在对输入列表inp_list进行一次卷积操作，然后将得到的结果进行切割。这是对GRU（门控循环单元）网络的优化，因为在标准的
            # GRU网络中，需要对上下文特征进行多次的卷积操作，这里提前一次性完成，提高了效率。
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            # 使用zip函数同时遍历inp_list和self.context_zqr_convs两个列表。在每次循环中，i是inp_list中的一个元素，conv是self.context_zqr_convs中的一个卷积层。
            # conv(i): 对inp_list中的元素i进行卷积操作。
            # .split(split_size=conv.out_channels//3, dim=1): 这个函数将卷积后的结果沿着通道维度（dim=1）进行切割，切割的大小是卷积层输出通道数的三分之一（conv.out_channels//3）。切割的结果是一个列表，其中每个元素都是切割后的一部分。
            # ------
            # 对于每一对(i, conv)（其中i是inp_list中的一个特征图，conv是self.context_zqr_convs中的一个卷积层），执行卷积操作conv(i)。
            # 使用.split(...)方法将卷积的结果沿着通道维度（dim=1）切分为三个部分。每部分的大小是总通道数的三分之一（split_size=conv.out_channels // 3）
            # 外部列表的长度与n_gru_layers一致，每个内部列表都包含三个部分，代表GRU的三个门的输出。
            # 对输入特征图进行卷积操作，并将结果切分为三个部分，以供GRU的三个门使用。
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                        for i, conv in zip(inp_list,
                                           self.context_zqr_convs)]  # list:3, 每个list里有3个tensor，list中的三个list的tensor分别为[1,128,136,240],[1,128,68,120],[1,128,34,60]

        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D  # in this branch
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius,
                             num_levels=self.args.corr_levels)  # fmap:[1,256,136,240];

        # coords:[1,2,136,240]，net_list[0]是最大的特征图list2，都是[1,128,136,240]
        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # [1,2,136,240]
            # """
            # 可视化一下coords1
            # """
            # from matplotlib import pyplot as plt
            # import numpy as np
            # plt.figure()
            # plt.subplot(121)
            # plt.title("{}".format(itr))
            # plt.imshow(np.array(coords1.cpu().numpy())[0][0])
            # plt.subplot(122)
            # plt.imshow(np.array(coords1.cpu().numpy())[0][1])
            # plt.show()

            corr = corr_fn(coords1)  # index correlation volume [1,36,136,240]

            flow = coords1 - coords0  # [1,2,136,240]  flow的第二通道是全0

            # from matplotlib import pyplot as plt
            # import numpy as np
            # plt.figure()
            # plt.subplot(121)
            # plt.title("{}".format(itr))
            # plt.imshow(np.array(flow.cpu().numpy())[0][0])
            # plt.subplot(122)
            # plt.imshow(np.array(flow.cpu().numpy())[0][1])
            # plt.show()

            with autocast(enabled=self.args.mixed_precision):  # mixed_precision:False
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False,
                                                 update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers == 3, iter16=True,
                                                 iter08=False, update=False)
                # n_gru_layers = 3
                # net_list:{list3}        [1,128,136,240][1,128,68,120][1,128,34,60]
                # inp_list:{list3}{list3} [1,128,136,240][1,128,68,120][1,128,34,60]
                # corr: [1,36,136,240]
                # flow: [1,2,136,240]
                net_list, up_mask, delta_flow = self.update_block(net=net_list, inp=inp_list, corr=corr, flow=flow,
                                                                  iter32=self.args.n_gru_layers == 3,
                                                                  iter16=self.args.n_gru_layers >= 2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow  # [1,2,136,240]

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up  # flow_up:[1,1,544,960]

        return flow_predictions
