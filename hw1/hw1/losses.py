import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        n_samples = x_scores.shape[0]
        indices = y.long().unsqueeze(-1)
        x_gt_scores = torch.gather(x_scores, -1, indices)
        x_scores_margin_loss_mat = x_scores - x_gt_scores + self.delta
        t_zero = torch.tensor([0.])
        x_scores_margin_loss_non_negative_mat = torch.where(x_scores_margin_loss_mat > 0,
                                                            x_scores_margin_loss_mat,
                                                            t_zero).reshape(x_scores.shape)
        redundant_deltas = self.delta * n_samples
        loss = (x_scores_margin_loss_non_negative_mat.sum() - redundant_deltas) / n_samples
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['x'] = x
        self.grad_ctx['mat_m'] = x_scores_margin_loss_mat
        self.grad_ctx['y_indices'] = indices
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx['x']
        x_scores_margin_loss_mat = self.grad_ctx["mat_m"]
        indices = self.grad_ctx['y_indices']
        t_zero = torch.tensor([0.])
        t_one = torch.tensor([1.])
        g_mat = torch.where(x_scores_margin_loss_mat > 0, t_one, t_zero)
        t = torch.zeros(g_mat.shape)
        t.scatter_(-1, indices, 1)
        g_temp = g_mat
        g_temp = g_temp.scatter_(-1, indices, 0).sum(1).unsqueeze(-1)
        n_classes = x_scores_margin_loss_mat.shape[-1]
        t_one = torch.ones( 1, n_classes)
        g_temp = g_temp * t_one
        g_mat = torch.where(t == 1, -g_temp, g_mat)
        n_samples = x_scores_margin_loss_mat.shape[0]
        grad = torch.mm(x.transpose(0, 1), g_mat) / n_samples

        # ========================
        return grad