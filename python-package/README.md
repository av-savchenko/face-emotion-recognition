# HSEmotions Python Library for Facial Emotion Recognition

## License

The code of HSEmotion Python Library is released under the Apache-2.0 License. There is no limitation for both academic and commercial usage.

## Installing

```
    python setup.py install
```

## Usage

```
    from hsemotion.facial_emotions import HSEmotionRecognizer
    model_name='enet_b0_8_best_afew'
    fer=HSEmotionRecognizer(model_name=model_name,device='cpu')
    emotion,scores=fer.predict_emotions(face_img,logits=True)
```

The following values of `model_name` parameter are supported:
- enet_b0_8_best_vgaf
- enet_b0_8_best_afew
- enet_b0_8_va_mtl
- enet_b2_8
- enet_b2_7

The method `predict_emotions` returns both the string value of predicted emotions (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, or Surprise) and scores at the output of the last layer. 
If the `logits` parameter is set to `True` (by default), the logits are returned, otherwise, the posterior probabilities are estimated from the logits using softmax.

In addition, it is possible to extract visual embeddings for classifier learning
```
    features=fer.extract_features(face_img)
```

The versions of these methods for a batch of images are also available
```
    emotions,scores=fer.predict_multi_emotions(face_img_list,logits=False)
    features=fer.extract_multi_features(face_img_list)
```

Complete usage example is available in the [test_hsemotion_package.ipynb](../src/test_hsemotion_package.ipynb)