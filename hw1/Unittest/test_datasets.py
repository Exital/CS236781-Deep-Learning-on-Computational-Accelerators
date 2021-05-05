import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import unittest
import hw1.datasets as hw1datasets
import cs236781.plot as plot

num_samples = 500
num_classes = 10
image_size = (3, 32, 32)

class test_datasets(unittest.TestCase):

    def test_first_test(self):
        image_shape = (3, 32, 64)
        num_classes = 3
        low, high = 0, 10

        # Generate some random images and check values
        X_ = None
        for i in range(100):
            X, y = hw1datasets.random_labelled_image(image_shape, num_classes, low, high)
            self.assertEqual(X.shape, image_shape)
            self.assertIsInstance(y, int)
            self.assertTrue(0 <= y < num_classes)
            self.assertTrue(torch.all((X >= low) & (X < high)))
            if X_ is not None:
                self.assertFalse(torch.all(X == X_))
            X_ = X

        plot.tensors_as_images([X, X_])
    def test_valueErr(self):
        num_samples = 500
        num_classes = 10
        image_size = (3, 32, 32)
        ds = hw1datasets.RandomImageDataset(num_samples, num_classes, *image_size)

        # You can load individual items from the dataset by indexing
        img0, cls0 = ds[139]

        # Plot first N images from the dataset with a helper function
        fig, axes = plot.dataset_first_n(ds, 9, show_classes=True, nrows=3)

        # The same image should be returned every time the same index is accessed
        for i in range(num_samples):
            X, y = ds[i]
            X_, y_ = ds[i]
            self.assertEqual(X.shape, image_size)
            self.assertIsInstance(y, int)
            self.assertEqual(y, y_)
            self.assertTrue(torch.all(X == X_))

        # Should raise if out of range
        for i in range(num_samples, num_samples + 10):
            with self.assertRaises(ValueError):
                ds[i]

    def test_iterator(self):
        ds = hw1datasets.ImageStreamDataset(num_classes, *image_size)

        # This dataset can't be indexed
        with self.assertRaises(NotImplementedError):
            ds[0]

        # There is no length
        with self.assertRaises(TypeError):
            len(ds)

        # Arbitrarily stop somewhere
        stop = torch.randint(2 ** 11, 2 ** 16, (1,)).item()

        # We can iterate over it, indefinitely
        for i, (X, y) in enumerate(ds):
            self.assertEqual(X.shape, image_size)
            self.assertIsInstance(y, int)

            if i > stop:
                break

        print(f'Generated {i} images')
        self.assertGreater(i, stop)
    def testIndexErr(self):
        import os
        import torchvision
        import torchvision.transforms as tvtf

        cfar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data_root = os.path.expanduser('~/.pytorch-datasets')

        cifar10_train_ds = torchvision.datasets.CIFAR10(
            root=data_root, download=True, train=True,
            transform=tvtf.ToTensor()
        )

        print('Number of samples:', len(cifar10_train_ds))

        # Plot them with a helper function
        fig, axes = plot.dataset_first_n(cifar10_train_ds, 64,
                                         show_classes=True, class_labels=cfar10_labels,
                                         nrows=8, hspace=0.5)

        subset_len = 5000
        subset_offset = 1234
        cifar10_train_subset_ds = hw1datasets.SubsetDataset(cifar10_train_ds, subset_len, subset_offset)

        dataset_x, dataset_y = cifar10_train_ds[subset_offset + 10]
        subset_x, subset_y = cifar10_train_subset_ds[10]

        # Tests
        self.assertEqual(len(cifar10_train_subset_ds), subset_len)
        self.assertTrue(torch.all(dataset_x == subset_x))
        self.assertEqual(dataset_y, subset_y)
        with self.assertRaises(IndexError, msg="Out of bounds index should raise IndexError"):
            tmp = cifar10_train_subset_ds[subset_len]