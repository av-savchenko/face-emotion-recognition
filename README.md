# HSEmotion (High-Speed face Emotion recognition) library
[![Downloads](https://static.pepy.tech/personalized-badge/hsemotion?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20installs)](https://pepy.tech/project/hsemotion)
[![pypi package](https://img.shields.io/badge/version-v0.2.0-blue)](https://pypi.org/project/hsemotion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classifying-emotions-and-engagement-in-online/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=classifying-emotions-and-engagement-in-online)

This repository contains code that was developed at the HSE University during the RSF (Russian Science Foundation) project no. 20-71-10010 (Efficient audiovisual analysis of dynamical changes in emotional state based on information-theoretic approach).

## News
- The paper "Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction" has been accepted as Oral talk at [ICML 2023](https://icml.cc/Conferences/2023). The source code to reproduce the results of this paper are available at this repository, see subsections "Adaptive Frame Rate" at [abaw3_train.ipynb](https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/src/ABAW/abaw3_train.ipynb) and [train_emotions-pytorch-afew-vgaf.ipynb](https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/src/affectnet/train_emotions-pytorch-afew-vgaf.ipynb)
- Our models let our team HSE-NN took the first place in the Learning from Synthetic Data (LSD) Challenge and the 3rd place in the Multi-Task Learning (MTL) Challenge in the [fourth Affective Behavior Analysis in-the-wild (ABAW) Competition](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/)
- Our models let our team HSE-NN took the 3rd place in the multi-task learning challenge, 4th places in Valence-Arousal and Expression challenges and 5th place in the Action Unite Detection Challenge in the [third Affective Behavior Analysis in-the-wild (ABAW) Competition](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/). Our approach is presented in the [paper](https://arxiv.org/abs/2203.13436) accepted at CVPR 2022 ABAW Workshop.

## Details
All the models were pre-trained for face identification task using [VGGFace2 dataset](https://github.com/ox-vgg/vgg_face2). In order to train PyTorch models, [SAM code](https://github.com/davda54/sam) was borrowed.

We upload several [models](models/affectnet_emotions) that obtained the state-of-the-art results for [AffectNet dataset](http://mohammadmahoor.com/affectnet/). The facial features extracted by these models lead to the state-of-the-art accuracy of face-only models on video datasets from EmotiW [2019](https://sites.google.com/view/emotiw2019), [2020](https://sites.google.com/view/emotiw2020) challenges: [AFEW (Acted Facial Expression In The Wild)](https://cs.anu.edu.au/few/AFEW.html), [VGAF (Video level Group AFfect)](https://ieeexplore.ieee.org/document/8925231),  [EngageWild](https://ieeexplore.ieee.org/document/8615851); and ABAW [CVPR 2022](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) and [ECCV 2022](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) challenges: Learning from Synthetic Data (LSD) and Multi-task Learning (MTL).

Here are the performance metrics (accuracy on AffectNet, AFEW and VGAF), F1-score on LSD, on the validation sets of the above-mentioned datasets and the mean inference time for our models on Samsung Fold 3 device with Qualcomm 888 CPU and Android 12:

| Model | AffectNet (8 classes)  | AffectNet (7 classes)   | AFEW  | VGAF  | LSD | MTL | Inference time, ms | Model size, MB
| :---:   | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [mobilenet_7.h5](models/affectnet_emotions/mobilenet_7.h5) | -  |  64.71   | 55.35 | 68.92  | - | 1.099 | 16 ± 5| 14 |
| [enet_b0_8_best_afew.pt](models/affectnet_emotions/enet_b0_8_best_afew.pt) | 60.95  | 64.63  | 59.89  | 66.80  | 59.32 | 1.110 |59 ± 26 | 16 |
| [enet_b0_8_best_vgaf.pt](models/affectnet_emotions/enet_b0_8_best_vgaf.pt) | 61.32   | 64.57   | 55.14  | 68.29  | 59.72 | 1.123 |59 ± 26 | 16 |
| [enet_b0_8_va_mtl.pt](models/affectnet_emotions/enet_b0_8_va_mtl.pt) | 61.93   | 64.94   | 56.73  | 66.58  | 60.94 | 1.276 |60 ± 32 | 16 |
| [enet_b0_7.pt](models/affectnet_emotions/enet_b0_7.pt) | -    | 65.74   | 56.99  | 65.18  | - | 1.111 |59 ± 26 | 16 | 16 |
| [enet_b2_7.pt](models/affectnet_emotions/enet_b2_7.pt) | -    | 66.34   | 59.63  | 69.84  | - | 1.134 |191 ± 18 | 30 |
| [enet_b2_8.pt](models/affectnet_emotions/enet_b2_8.pt) | 63.03  | 66.29 | 57.78  | 70.23  | 52.06 | 1.147 |191 ± 18 | 30 |
| [enet_b2_8_best.pt](models/affectnet_emotions/enet_b2_8_best.pt) | 63.125  | 66.51 | 56.73  | 71.12  | - | - |191 ± 18 | 30 |

Please note, that we report the accuracies for AFEW and VGAF only on the subsets, in which MTCNN detects facial regions. The code contains also computation of overall accuracy on the complete testing set, which is slightly lower due to the absence of faces or failed face detection.

## Usage
Special python packages [hsemotion](https://github.com/HSE-asavchenko/hsemotion) and [hsemotion-onnx](https://github.com/HSE-asavchenko/hsemotion-onnx) were prepared to simplify the usage of our models for face expression recognition and extraction of visual emotional embeddings. They can be installed via pip:
```
    pip install hsemotion
    pip install hsemotion-onnx
```

In order to run our code on the datasets, please prepare them firstly using our TensorFlow notebooks: [train_emotions.ipynb](src/affectnet/train_emotions.ipynb), [AFEW_train.ipynb](src/AFEW_train.ipynb) and [VGAF_train.ipynb](src/VGAF_train.ipynb).

If you want to run our mobile application, please, run the following scripts inside [mobile_app](mobile_app) folder:
```
python to_tflite.py
python to_pytorchlite.py
```

NOTE!!! I updated the models so that they should work with recent timm library. However, for v0.1 version, please be sure that EfficientNet models for PyTorch are based on old timm 0.4.5 package, so that exactly this version should be installed by the following command:
```
pip install timm==0.4.5
```


## Research papers

If you use our models, please cite the following papers:
```BibTex
@inproceedings{savchenko2023facial,
  title = 	 {Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction},
  author =       {Savchenko, Andrey},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  pages = 	 {30119--30129},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  url={https://proceedings.mlr.press/v202/savchenko23a.html}
}
```

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
@inproceedings{Savchenko_2022_ECCVW,
  author    = {Savchenko, Andrey V.},
  title     = {{MT-EmotiEffNet} for Multi-task Human Affective Behavior Analysis and Learning from Synthetic Data},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV 2022) Workshops},
  pages={45--59},
  year={2023},
  organization={Springer},
  url={https://arxiv.org/abs/2207.09508}
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
