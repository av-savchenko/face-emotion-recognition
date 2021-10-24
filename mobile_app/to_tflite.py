#https://www.tensorflow.org/lite/performance/post_training_quantization
#https://gist.github.com/NobuoTsukamoto/b42128104531a7612e5c85e246cb2dac
import sys,os
import tensorflow as tf

def convert_pb(input_model_file, quantize=False):
    output_model_file=os.path.splitext(input_model_file)[0]
    if quantize:
        output_model_file+='_quant'
    
    print(output_model_file)
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = input_model_file, 
        input_arrays = ['input_1'],
        input_shapes={'input_1':[1,224,224,3]},
        output_arrays = ['age_pred/Softmax','gender_pred/Sigmoid','ethnicity_pred/Softmax','global_pooling/Mean'] 
    )
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(output_model_file+'.tflite', 'wb') as f:
        f.write(tflite_model)

def convert_h5(input_model_file, output_model_file, quantize=False):
    #output_model_file=os.path.splitext(input_model77_file)[0]
    if quantize:
        output_model_file+='_quant8'
    print(output_model_file)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file( input_model_file)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.float16]
    tfmodel = converter.convert()
    with open (output_model_file+'.tflite', "wb") as f:
        f.write(tfmodel)
      
if __name__ == '__main__':
    convert_pb('app/src/main/assets/age_gender_ethnicity_224_deep-03-0.13-0.97-0.88.pb', quantize=False)
    convert_h5('../models/affectnet_emotions/mobilenet_7.h5', 'app/src/main/assets/emotions_mobilenet',quantize=False)
