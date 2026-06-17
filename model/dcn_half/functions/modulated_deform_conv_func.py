#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN2 as DCN

class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        # Align weight/bias dtype to input dtype (required by CUDA kernel)
        input_dtype = input.dtype
        if weight.dtype != input_dtype:
            weight = weight.to(input_dtype)
        if bias.dtype != input_dtype:
            bias = bias.to(input_dtype)

        output = DCN.modulated_deform_conv_forward(input.contiguous(), weight.contiguous(), bias.contiguous(),
                                         offset.contiguous(), mask.contiguous(),
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.groups,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors

        # Align grad_output dtype to weight dtype (for FP16 training)
        target_dtype = weight.dtype
        if grad_output.dtype != target_dtype:
            grad_output = grad_output.to(target_dtype)

        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            DCN.modulated_deform_conv_backward(input.contiguous(), weight.contiguous(),
                                     bias.contiguous(),
                                     offset.contiguous(), mask.contiguous(),
                                     grad_output.contiguous(),
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.groups,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None, None, None
