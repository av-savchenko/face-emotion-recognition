This repository contains code of face emotion recognition that was developed in the RSF (Russian Science Foundation) project no. 20-71-10010 (Efficient audiovisual analysis of dynamical changes in emotional state based on information-theoretic approach).

If you use our models, please cite the following papers:
```BibTex
@inproceedings{savchenko2021facial,
  title={Facial expression and attributes recognition based on multi-task learning of lightweight neural networks},
  author={Savchenko, Andrey V.},
  booktitle={Proceedings of the 19th International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={119--124},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@article{savchenko2022classifying,
  title={Classifying emotions and engagement in online learning based on a single facial expression recognition neural network},
  author={Savchenko, Andrey V and Savchenko, Lyudmila V and Makarov, Ilya},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/9815154}
}
```

**[News]** Our models let our team HSE-NN took the 3rd place in the multi-task learning challenge, 4th places in Valence-Arousal and Expression challenges and 5th place in the Action Unite Detection Challenge in the [third Affective Behavior Analysis in-the-wild (ABAW) Competition](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/). Our approach is presented in the [paper](https://arxiv.org/abs/2203.13436) accepted at CVPR 2022 ABAW Workshop.

All the models were pre-trained for face identification task using [VGGFace2 dataset](https://github.com/ox-vgg/vgg_face2). In order to train PyTorch models, [SAM code](https://github.com/davda54/sam) was borrowed.

We upload several [models](models/affectnet_emotions) that obtained the state-of-the-art results for [AffectNet dataset](http://mohammadmahoor.com/affectnet/). The facial features extracted by these models lead to the state-of-the-art accuracy of face-only models on video datasets from EmotiW [2019](https://sites.google.com/view/emotiw2019), [2020](https://sites.google.com/view/emotiw2020) challenges: [AFEW (Acted Facial Expression In The Wild)](https://cs.anu.edu.au/few/AFEW.html), [VGAF (Video level Group AFfect)](https://ieeexplore.ieee.org/document/8925231) and [EngageWild](https://ieeexplore.ieee.org/document/8615851).

Here are the accuracies measure on the testing set of above-mentioned datasets:

| Model | AffectNet (8 classes), original | AffectNet (8 classes), aligned  | AffectNet (7 classes), original   | AffectNet (7 classes), aligned   | AFEW  | VGAF  |
| :---:   | :-: | :-:  | :-: | :-: | :-: | :-: |
| [mobilenet_7.h5](models/affectnet_emotions/mobilenet_7.h5) | -  |  -  | 64.71   |  -  | 55.35 | 68.92  |
| [enet_b0_8_best_afew.pt](models/affectnet_emotions/enet_b0_8_best_afew.pt) | 60.95  | 60.18  | 64.63  | 64.54   | 59.89  | 66.80  |
| [enet_b0_8_best_vgaf.pt](models/affectnet_emotions/enet_b0_8_best_vgaf.pt) | 61.32   | 61.03  | 64.57   | 64.89   | 55.14  | 68.29  |
| [enet_b0_8_va_mtl.pt](models/affectnet_emotions/enet_b0_8_va_mtl.pt) | 61.93   | -  | 64.94   | -   | 56.73  | 66.58  |
| [enet_b0_7.pt](models/affectnet_emotions/enet_b0_7.pt) | -    | - | 65.74   | 65.74   | 56.99  | 65.18  |
| [enet_b2_8.pt](models/affectnet_emotions/enet_b2_8.pt) | 63.025  | 62.40  | 66.29 | -   | 57.78  | 70.23  |
| [enet_b2_7.pt](models/affectnet_emotions/enet_b2_7.pt) | -    | - | 65.91   | 66.34   | 59.63  | 69.84  |

Please note, that we report the accuracies for AFEW and VGAFonly on the subsets, in which MTCNN detects facial regions. The code contains also computation of overall accuracy on the complete testing set, which is slightly lower due to the absence of faces or failed face detection.

In order to run our code on the datasets, please prepare them firstly using our TensorFlow notebooks: [train_emotions.ipynb](src/train_emotions.ipynb), [AFEW_train.ipynb](src/AFEW_train.ipynb) and [VGAF_train.ipynb](src/VGAF_train.ipynb).

If you want to run our mobile application, please, run the following scripts inside [mobile_app](mobile_app) folder:
```
python to_tflite.py
python to_pytorchlite.py
```

NOTE!!! I updated the models so that they should work with recent timm library. However, for v0.1 version, please be sure that EfficientNet models for PyTorch are based on old timm 0.4.5 package, so that exactly tis version should be installed by the following command:
```
pip install timm==0.4.5
```
