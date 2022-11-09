# HSEmotionONNX Python Library for Facial Emotion Recognition

## License

The code of HSEmotionONNX Python Library is released under the Apache-2.0 License. There is no limitation for both academic and commercial usage.

## Installing

```
    python setup.py install
```


## Usage

```
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    model_name='enet_b0_8_best_afew'
    fer=HSEmotionRecognizer(model_name=model_name)
    emotion,scores=fer.predict_emotions(face_img,logits=False)
```

The following values of `model_name` parameter are supported:
- enet_b0_8_best_vgaf
- enet_b0_8_best_afew
- enet_b0_8_va_mtl
- enet_b2_8
- enet_b2_7

The method `predict_emotions` returns both the string value of predicted emotions (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, or Surprise) and scores at the output of the last layer. 
If the `logits` parameter is set to `True` (by default), the logits are returned, otherwise, the posterior probabilities are estimated from the logits using softmax.


The versions of this method for a batch of images are also available
```
    emotions,scores=fer.predict_multi_emotions(face_img_list,logits=False)
```

The usage example is available in the [onnx folder](../../src/onnx)
