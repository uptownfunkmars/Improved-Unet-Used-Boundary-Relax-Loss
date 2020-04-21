import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage.interpolation import shift

from skimage.segmentation import find_boundaries


# class RelaxedBoundaryLossToTensor(object):
#     """
#     Boundary Relaxation
#     """
#
#     def __init__(self, ignore_id, num_classes):
#         self.ignore_id = ignore_id
#         self.num_classes = num_classes
#
#     def new_one_hot_converter(self, a):
#         ncols = self.num_classes + 1
#         out = np.zeros((a.size, ncols), dtype=np.uint8)
#         out[np.arange(a.size), a.ravel()] = 1
#         out.shape = a.shape + (ncols,)
#         return out
#
#     def __call__(self, img):
#
#         img_arr = np.array(img)
#         img_arr[img_arr == self.ignore_id] = self.num_classes
#
#         # if cfg.STRICTBORDERCLASS != None:
#         #     one_hot_orig = self.new_one_hot_converter(img_arr)
#         #     mask = np.zeros((img_arr.shape[0], img_arr.shape[1]))
#         #     for cls in cfg.STRICTBORDERCLASS:
#         #         mask = np.logical_or(mask, (img_arr == cls))
#         one_hot = 0
#
#         border = 1
#         # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
#         #     border = border // 2
#         #     border_prediction = find_boundaries(img_arr, mode='thick').astype(np.uint8)
#
#         for i in range(-border, border + 1):
#             for j in range(-border, border + 1):
#                 shifted = shift(img_arr, (i, j), cval=self.num_classes)
#                 one_hot += self.new_one_hot_converter(shifted)
#
#         one_hot[one_hot > 1] = 1
#
#         # if cfg.STRICTBORDERCLASS != None:
#         #     one_hot = np.where(np.expand_dims(mask, 2), one_hot_orig, one_hot)
#
#         one_hot = np.moveaxis(one_hot, -1, 0)
#
#         # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
#         #     one_hot = np.where(border_prediction, 2 * one_hot, 1 * one_hot)
#         #     # print(one_hot.shape)
#         return torch.from_numpy(one_hot).byte()


def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    # print((multihotmask * (soft * multihotmask).sum(1, keepdim=True)).size())
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        # self.batch_weights = cfg.BATCH_WEIGHTING
        self.batch_weights = True
        self.fp16 = False


    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
        #     border_weights = 1 / border_weights
        #     target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss

