import torch
import torch.nn as nn


class AsymmetricLossWithWeight(nn.Module):
    '''
    AsymmetricLoss * balance_weight
    
    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossWithWeight, self).__init__()
        self.register_buffer('_LABEL_NUMS', torch.tensor([525, 215, 211, 161, 77, 174, 964]).float())
        self.register_buffer('ALPHA', (self._LABEL_NUMS.sum() - self._LABEL_NUMS) / self._LABEL_NUMS.sum())
        self.ALPHA = self.ALPHA / self.ALPHA.max()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        self.xs_pos = self.xs_pos * self.targets
        self.xs_neg = self.xs_neg * self.anti_targets
        self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                      self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        self.loss *= self.asymmetric_w
        self.loss *= self.ALPHA

        return -self.loss.sum()


if __name__ == '__main__':
    criterion = AsymmetricLossWithWeight()
    # x = torch.randn(5, 7) * 10
    x = torch.tensor(
        [
            [-1, 1, 1, 1, 1, 1, 1],
            [-1, 1, 1, 1, 1, 1, 1],
            [1, 1, -1, -1, 1, -1, 1],
            [1, 1, 1, -1, 1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    ).float()
    x *= 1e6
    y = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).float()
    loss = criterion(x, y)
    print(loss)