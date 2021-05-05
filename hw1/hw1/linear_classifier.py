import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader
import cs236781.dataloader_utils as dataloader_utils
from .losses import ClassifierLoss, SVMHingeLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.randn(n_features, n_classes) * weight_std
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)  # X*W
        y_pred = class_scores.argmax(dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        num_of_accurate_predictions = ((y - y_pred) == 0).sum().item()
        num_of_samples = y.shape[0]
        acc = num_of_accurate_predictions / num_of_samples
        # ========================

        return acc * 100

    def train(
            self,
            dl_train: DataLoader,
            dl_valid: DataLoader,
            loss_fn: ClassifierLoss,
            learn_rate=0.1,
            weight_decay=0.001,
            max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            # print(f'{epoch_idx}.')
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            results = (train_res, valid_res)
            train_idx, valid_idx = 0, 1
            epoc_weights = self.weights
            for idx, dl in enumerate([dl_train, dl_valid]):
                # predict data using current weights
                x, y = dataloader_utils.flatten(dl)
                y_pred, x_scores = self.predict(x)

                # calculate accuracy
                accuracy = self.evaluate_accuracy(y, y_pred)

                # calulate mean pointwise data dependent loss and gradient
                pw_loss = loss_fn(x, y, x_scores, y_pred)
                pw_grad = loss_fn.grad()

                # calculate regularization loss and gradient
                reg_loss = weight_decay * (torch.norm(epoc_weights).item() ** 2) / 2
                reg_grad = weight_decay * epoc_weights

                # sum it
                loss = pw_loss + reg_loss
                grad = pw_grad + reg_grad

                # weights update (train_dl case only)
                if idx == train_idx:
                    self.weights -= learn_rate * grad

                results[idx][0].append(accuracy)
                results[idx][1].append(loss)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if has_bias == True:
            t_weights = self.weights[1:]
        else:
            t_weights = self.weights

        w_images = t_weights.transpose(0, 1).reshape(self.n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0, learn_rate=0, weight_decay=0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.008
    hp['weight_decay'] = 0.2
    # ========================

    return hp
