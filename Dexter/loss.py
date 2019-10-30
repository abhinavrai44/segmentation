from __future__ import division
import numpy as np
import torch
from torch.nn import functional as F

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    # Bootstrap Loss
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """
    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = loss_pos_pix
    loss_neg = loss_neg_pix

    loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg
    final_loss = 0.0
    for i in range(label.size()[0]):
        top_k = torch.flatten(loss[i])
        top_k, _ = top_k.topk(64*512)
        final_loss = final_loss + top_k.sum()

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss
    

# def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
#     # Focal Loss
#     """Define the class balanced cross entropy loss to train the network
#     Args:
#     output: Output of the network
#     label: Ground truth label
#     size_average: return per-element (pixel) average loss
#     batch_average: return per-batch average loss
#     void_pixels: pixels to ignore from the loss
#     Returns:
#     Tensor that evaluates the loss
#     """
#     assert(output.size() == label.size())

#     gamma = 2.0
#     eps = 2e-6

#     labels = torch.ge(label, 0.5).float()

#     num_labels_pos = torch.sum(labels)
#     num_labels_neg = torch.sum(1.0 - labels)
#     num_total = num_labels_pos + num_labels_neg

#     # output_gt_zero = torch.ge(output, 0).float()
#     # loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
#     #     1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))


#     # loss_pos_pix = -torch.mul(labels, loss_val)
#     # loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

#     # if void_pixels is not None:
#     #     w_void = torch.le(void_pixels, 0.5).float()
#     #     loss_pos_pix = torch.mul(w_void, loss_pos_pix)
#     #     loss_neg_pix = torch.mul(w_void, loss_neg_pix)
#     #     num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

#     pos_pt = torch.where(label>=0.5 , output, torch.ones_like(output))
#     neg_pt = torch.where(label<0.5 , output, torch.zeros_like(output))

#     pos_modulating = (1-pos_pt)**gamma
#     neg_modulating = (neg_pt)**gamma

#     pos = -(num_labels_neg / num_total) * pos_modulating*torch.log(pos_pt)
#     neg = -(num_labels_pos / num_total) * neg_modulating*torch.log(1-neg_pt)

#     final_loss = pos + neg 
#     final_loss = final_loss.sum()

#     if size_average:
#         final_loss /= np.prod(label.size())
#     elif batch_average:
#         final_loss /= label.size()[0]

#     return final_loss


# def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
#     """Define the class balanced cross entropy loss to train the network
#     Args:
#     output: Output of the network
#     label: Ground truth label
#     size_average: return per-element (pixel) average loss
#     batch_average: return per-batch average loss
#     void_pixels: pixels to ignore from the loss
#     Returns:
#     Tensor that evaluates the loss
#     """
#     assert(output.size() == label.size())

#     labels = torch.ge(label, 0.5).float()

#     num_labels_pos = torch.sum(labels)
#     num_labels_neg = torch.sum(1.0 - labels)
#     num_total = num_labels_pos + num_labels_neg

#     output_gt_zero = torch.ge(output, 0).float()
#     loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
#         1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

#     loss_pos_pix = -torch.mul(labels, loss_val)
#     loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

#     if void_pixels is not None:
#         w_void = torch.le(void_pixels, 0.5).float()
#         loss_pos_pix = torch.mul(w_void, loss_pos_pix)
#         loss_neg_pix = torch.mul(w_void, loss_neg_pix)
#         num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

#     loss_pos = torch.sum(loss_pos_pix)
#     loss_neg = torch.sum(loss_neg_pix)

#     final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

#     if size_average:
#         final_loss /= np.prod(label.size())
#     elif batch_average:
#         final_loss /= label.size()[0]

#     return final_loss
