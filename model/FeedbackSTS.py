import torch
import torch.nn as nn
import torch.nn.functional as F

#from .dcn.modules.deform_conv import DeformConv
from .dcn_half.modules.deform_conv import DeformConv

DEFAULT_SEQUENCE_LENGTH = 11
DEFAULT_SPARSE_INTERVAL = 2

class BFBM(nn.Module):
    """
    Basic Feedback Module (BFBM)
    Fig . 2(a) (b) (c)
    Including: 
    - Feature Extraction (Pyramid downsampling)
    - Feature Alignment (Pyramid deformable alignment)
    """
    def __init__(self, num_feat=64, deformable_groups=4, level_num=3):
        super(BFBM, self).__init__()
        assert level_num > 1, "level_num must > 1"
        self.level_num = level_num

        # ---- Feature Extraction (Fig. b) ----
        # MyNet16 computes separate pyramids for the neighbor and reference.
        # Keep the FeedbackSTS class names, but preserve those exact weights.
        self.conv_nbr_list = nn.ModuleList()
        self.conv_ref_list = nn.ModuleList()
        for _ in range(level_num - 1):
            self.conv_nbr_list.append(
                nn.Conv2d(num_feat, num_feat, 3, stride=2, padding=1)
            )
            self.conv_nbr_list.append(
                nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
            )
            self.conv_ref_list.append(
                nn.Conv2d(num_feat, num_feat, 3, stride=2, padding=1)
            )
            self.conv_ref_list.append(
                nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
            )

        # ---- Feature Alignment (Fig. c) ----
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(level_num, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == level_num:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DeformConv(
                num_feat, num_feat, 3, padding=1, stride=1,
                deformable_groups=deformable_groups
            )
            if i < level_num:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # DCN backward returns a view in PyTorch 2.x; in-place ReLU would
        # overwrite it and invalidate the custom autograd function.
        self.lrelu = nn.ReLU(inplace=False)

    def feature_extraction(self, x, convolutions):
        """
        Fig. 2(b): Generate a pyramid feature list, where index 0 is 
        the original size, and as the index increases, the size halves.
        """
        feat_list = [x]
        current = x
        for i in range(self.level_num - 1):
            current = self.lrelu(convolutions[2 * i](current))
            current = self.lrelu(convolutions[2 * i + 1](current))
            feat_list.append(current)
        return feat_list

    def feature_alignment(self, nbr_feat_list, ref_feat_list):
        """
        Fig. 2(c): Pyramid deformable alignment
        returning the aligned feature at the original size.
        """
        upsampled_offset = None
        upsampled_feat = None
        for i in range(self.level_num, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_list[i-1], ref_feat_list[i-1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == self.level_num:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_list[i-1].contiguous(), 
                                        offset.contiguous())

            if i < self.level_num:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                feat = self.lrelu(feat)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        return feat

    def forward(self, neighbor, reference):
        """
        Args:
            neighbor (Tensor): current frame features (B, C, H, W)
            reference (Tensor): reference frame features (B, C, H, W)
        Returns:
            aligned (Tensor): aligned frame features (B, C, H, W)
        """
        nbr_feats = self.feature_extraction(neighbor, self.conv_nbr_list)
        ref_feats = self.feature_extraction(reference, self.conv_ref_list)
        aligned = self.feature_alignment(nbr_feats, ref_feats)
        return aligned


class SSM(nn.Module):
    """
    Sparse Semantic Module (SSM)
        1. Sparse Grouping
        2. Intra-Group Spatio-Temporal Semantic Propagation
        3. Temporal Reassembly
    """
    def __init__(self, num_feat=64, deformable_groups=4, level_num=3, 
                 t=1, isForward=True):
        """
        Args:
            t (int): sparse sampling interval T, i.e., the number of groups.
            isForward (bool): True = forward propagation (from past to future), False = backward propagation (from future to past)
        """
        super(SSM, self).__init__()
        self.t = t
        self.isForward = isForward
        self.bfbm = BFBM(num_feat=num_feat, deformable_groups=deformable_groups, level_num=level_num)

    def forward(self, X):
        """
        X: (B, C, D, H, W)   D = Frame Num
        Returns: (B, C, D, H, W)
        """
        _, _, D, _, _ = X.shape
        assert D > 1, "Need more than 1 frame"

        # Store the processed features of each group 
        # and reassemble them according to the original indices
        output_feats = [None] * D

        # 1. Sparse Grouping: divide into t groups with an interval of t.
        for group_id in range(self.t):
            indices = list(range(group_id, D, self.t))
            if not indices:
                continue

            # 2. Intra-Group Propagation
            if self.isForward:
                # Forward direction: starting from the first frame within the group.
                prop_feat = X[:, :, indices[0], :, :]
                output_feats[indices[0]] = prop_feat
                for i in range(1, len(indices)):
                    curr_feat = X[:, :, indices[i], :, :]
                    # MyNet16 aligns the propagated history feature to the
                    # current feature. Preserve that direction exactly.
                    prop_feat = self.bfbm(
                        neighbor=prop_feat,
                        reference=curr_feat,
                    )
                    output_feats[indices[i]] = prop_feat
            else:
                # Backward direction: starting from the last frame within the group.
                prop_feat = X[:, :, indices[-1], :, :]
                output_feats[indices[-1]] = prop_feat
                for i in range(len(indices)-2, -1, -1):
                    curr_feat = X[:, :, indices[i], :, :]
                    prop_feat = self.bfbm(
                        neighbor=prop_feat,
                        reference=curr_feat,
                    )
                    output_feats[indices[i]] = prop_feat

        # 3. Temporal Reassembly: stack in the original order
        X_out = torch.stack(output_feats, dim=2)
        return X_out

class STSRM(nn.Module):
    """
    Spatio-Temporal Semantic Feedback Strategy  (STSRM)

    isForward = True, Forward Spatio-Temporal Semantic Feedback Strategy (FSTSRM)
    isForward = False, Backward Spatio-Temporal Semantic Feedback Strategy (BSTSRM)
    """
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, 
                 ssm_level_num = 3, t = 1, 
                 isForward = True, using_ssm = True):
        super(STSRM, self).__init__()

        self.conv1 = nn.Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.batch_norm1 = nn.BatchNorm3d(out_feat)
        self.conv2 = nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.batch_norm2 = nn.BatchNorm3d(out_feat)

        ## Res Part
        self.using_ssm = using_ssm
        ## Using Pda
        self.conv3 = nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)
        self.ssm = SSM(num_feat = out_feat, level_num = ssm_level_num,
                      t = t, isForward = isForward)
        ## no pda
        self.conv1_no_ssm = nn.Conv3d(inp_feat, out_feat, kernel_size=3, stride= 1, padding = 1, bias=True)
        self.conv2_no_ssm = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        #keep input as a residual
        res = x
        x = self.batch_norm1(self.conv1(x))
        x = F.relu(x)
        x = self.batch_norm2(self.conv2(x))
        x = F.relu(x)

        #sum output and residual
        if (self.using_ssm):
            res = self.conv3(res)
            res = self.ssm(res)
        else:
            res = self.conv1_no_ssm(res)
            res = self.conv2_no_ssm(res)
        return x + res

#Upsampler block to reconstruct image
class Up_3D(nn.Module):
    """
    Upscale 3D Block
    """
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Up_3D, self).__init__()
        
        self.deconv =  nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel), 
                                       stride=(1,stride,stride), padding=(0, padding, padding), 
                                       output_padding=0, bias=True)
    
    def forward(self, x):
        return F.relu(self.deconv(x))

class FeedbackSTS(nn.Module):
    def __init__(self,
                 inch=1,
                 outch=1,
                 feat_channels=[8, 16, 32, 64, 128],
                 ssm_level_nums=[2, 2, 2, 2, 2],
                 t_s=None,
                 down_using_ssm=[True, True, True, True, True],
                 up_using_ssm=[True, True, True, True],
                 down_is_forward=[True, True, True, True, True],
                 up_is_forward=[False, False, False, False]
                 ):
        super(FeedbackSTS, self).__init__()

        # Paper configuration: sequence length 11 and sparse-frame interval
        # T=2 at every encoder/decoder level. Keep it explicit so training,
        # testing, and checkpoint evaluation cannot silently disagree.
        if t_s is None:
            t_s = [DEFAULT_SPARSE_INTERVAL] * len(feat_channels)
        elif isinstance(t_s, int):
            t_s = [t_s] * len(feat_channels)
        if len(t_s) != len(feat_channels):
            raise ValueError(
                f"t_s must contain {len(feat_channels)} values, got {len(t_s)}"
            )
        if any(int(t) < 1 for t in t_s):
            raise ValueError(f"Every sparse-frame interval must be >= 1, got {t_s}")
        self.t_s = tuple(int(t) for t in t_s)

        # Encoder part: each stage consists of STSRM followed by MaxPool3d (the last stage has no pooling).
        self.enc_convs = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        for i in range(len(feat_channels)):
            inp = inch if i == 0 else feat_channels[i-1]
            self.enc_convs.append(
                STSRM(inp_feat=inp, out_feat=feat_channels[i],
                      t=t_s[i], ssm_level_num=ssm_level_nums[i],
                      using_ssm=down_using_ssm[i], isForward=down_is_forward[i])
            )
            # No pooling is applied after the last stage.
            if i < len(feat_channels) - 1:
                self.enc_pools.append(nn.MaxPool3d((1, 2, 2)))
            else:
                self.enc_pools.append(None)  # Placeholder, skip in forward.

        # Decoder Part
        self.dec_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(len(feat_channels)-1):
            dec_idx = len(feat_channels)-1 - i
            # Upsampling
            self.up_blocks.append(Up_3D(feat_channels[dec_idx], feat_channels[dec_idx-1]))
            # Decoder Convolution
            self.dec_convs.append(
                STSRM(inp_feat=2 * feat_channels[dec_idx-1], out_feat=feat_channels[dec_idx-1],
                      t=t_s[dec_idx-1], ssm_level_num=ssm_level_nums[dec_idx-1],
                      using_ssm=up_using_ssm[dec_idx-1], isForward=up_is_forward[dec_idx-1])
            )

        # Final layer
        self.final = nn.Conv3d(feat_channels[0], outch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        shape = x.shape
        assert len(shape) == 5, "The input tensor should be 5-dimensional."

        # Encoder forward: store the output of each layer for skip connections.
        enc_outputs = []
        cur = x
        for i, conv in enumerate(self.enc_convs):
            cur = conv(cur)
            enc_outputs.append(cur)
            if self.enc_pools[i] is not None:
                cur = self.enc_pools[i](cur)

        # base is the output of the last layer (without pooling).
        base = enc_outputs[-1]

        # Decoder forward
        dec_cur = base
        for i, (up_block, dec_conv) in enumerate(zip(self.up_blocks, self.dec_convs)):
            # Upscaling
            up = up_block(dec_cur)
            skip = enc_outputs[-2 - i]
            # concat
            concat = torch.cat([up, skip], dim=1)
            dec_cur = dec_conv(concat)

        seg = self.final(dec_cur)
        seg = torch.sigmoid(seg)
        return seg
