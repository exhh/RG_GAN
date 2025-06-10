#Region-guided generative adversarial network (RG-GAN) for lesion detection in PET images

This code is used to detect neuroendocrine tumors/lesions on positron emission tomorgraphy (PET) images with a region-guided GAN model. The code is implemented with PyTorch (version 1.10.1, https://pytorch.org/) on a Ubuntu Linux machine.

Usage (please refer to the paper below for the details of data format):

For training: ./train.sh

For testing: ./eval.sh



Reference:
1. Xinyi Yang, Bennett B. Chin, Michael Silosky, Jonathan Wehrend, Daniel V. Litwiller, Debashis Ghosh, and Fuyong Xing, "Learning Without Real Data Annotations to Detect Hepatic Lesions in PET Images", IEEE Transactions on Biomedical Engineering, vol. 71, no. 2, pp. 679 - 688, 2024.
