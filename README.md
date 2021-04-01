This repository contains code of face emotion recognition that was developed in the RSF (Russian Science Foundation) project no. 20-71-10010 (Efficient audiovisual analysis of dynamical changes in emotional state based on information-theoretic approach).

Our approach is described in the [arXiv paper](https://arxiv.org/abs/2103.17107)

All the models were pre-trained for face identification task using [VGGFace2 dataset](https://github.com/ox-vgg/vgg_face2). In order to train PyTorch models, [SAM code](https://github.com/davda54/sam) was borrowed.

We upload several [models](models/affectnet_emotions) that obtained the state-of-the-art results for [AffectNet dataset](http://mohammadmahoor.com/affectnet/). The facial features extracted by these models lead to the state-of-the-art accuracy of face-only models on video datasets from EmotiW [2019](https://sites.google.com/view/emotiw2019), [2020](https://sites.google.com/view/emotiw2020) challenges: [AFEW (Acted Facial Expression In The Wild)](https://cs.anu.edu.au/few/AFEW.html) and [VGAF (Video level Group AFfect)](https://ieeexplore.ieee.org/document/8925231).

Here are the accuracies measure on the testing set of above-mentioned datasets:

| Model | AffectNet (8 classes)  | AffectNet (7 classes)   | AFEW  | VGAF  |
| :---:   | :-: | :-: | :-: | :-: |
| [mobilenet_7.h5](models/affectnet_emotions/mobilenet_7.h5) | -  | 64.71   | 55.35  | 70.31  |
| [enet_b0_8_best_afew.pt](models/affectnet_emotions/enet_b0_8_best_afew.pt) | 60.95  | 64.63   | 59.89  | 66.80  |
| [enet_b0_8_best_vgaf.pt](models/affectnet_emotions/enet_b0_8_best_vgaf.pt) | 61.32  | 64.57   | 55.14  | 68.29  |
| [enet_b0_7.pt](models/affectnet_emotions/enet_b0_7.pt) | -  | 65.74   | 56.99  | 65.18  |

Please note, that we report the accuracies for AFEW and VGAFonly on the subsets, in which MTCNN detects facial regions. The code contains also computation of overall accuracy on the complete testing set, which is slightly lower due to the absence of faces or failed face detection.

In order to run our code on the datasets, please prepare them firstly using our TensorFlow notebooks: [train_emotions.ipynb](src/train_emotions.ipynb), [AFEW_train.ipynb](src/AFEW_train.ipynb) and [VGAF_train.ipynb](src/VGAF_train.ipynb).
