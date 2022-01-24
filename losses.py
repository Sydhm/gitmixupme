import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss(reduction='none')


    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_tensor):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        # bs = prediction.size(0)
        if isinstance(target_tensor, bool):
            target_tensor = self.get_target_tensor(prediction, target_tensor)
        loss = self.loss(prediction, target_tensor)

        return loss

# def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coeff(input[i, ...], target[i, ...])
#         return dice / input.shape[0]


# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all classes
#     assert input.size() == target.size()
#     dice = 0
#     for channel in range(input.shape[1]):
#         dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

#     return dice / input.shape[1]


# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     assert input.size() == target.size()
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)

# def task_loss(input, target, multiclass = True):

#     loss =  torch.nn.CrossEntropyLoss()(input, target.permute(0,2,3,1).argmax(-1)) \
#            + dice_loss(torch.nn.functional.softmax(input, dim=1).float(),
#                        target.float(),
#                        multiclass=True)
#     return loss

def _softmax_weighted_loss(logits, gt):
    """
    Calculate weighted cross-entropy loss.
    """
    softmaxpred = nn.Softmax(1)(logits)
    for i in range(5):
        gti = gt[:,i,:,:]
        predi = softmaxpred[:,i,:,:]
        weighted = 1-(torch.sum(gti) / torch.sum(gt))
        if i == 0:

            raw_loss = -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
            #raw_loss = -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))

    loss = torch.mean(raw_loss)
    # print(loss)
    return loss

def _dice_loss_fun(logits, gt):
    """
    Calculate dice loss.
    """
    dice = 0
    eps = 1e-7
    softmaxpred = nn.Softmax(1)(logits)
    for i in range(5):
        inse = torch.sum(softmaxpred[:, i, :, :]*gt[:, i, :, :])
        l = torch.sum(softmaxpred[:, i, :, :]*softmaxpred[:, i, :, :])
        r = torch.sum(gt[:, i, :, :])
        dice += 2.0 * inse/(l+r+eps)

    return 1 - 1.0 * dice / 5


def task_loss(prediction, gt):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt)
    dice_loss = _dice_loss_fun(prediction, gt)

    return ce_loss + dice_loss