import numpy as np
import torch,timm
import onnx
import onnxruntime as ort

def convert_pt_to_onnx(model_name, img_size=224):
    print('Processing',model_name,img_size)
    dummy_input = torch.randn(1, 3, img_size, img_size)
    model=torch.load(f'../models/affectnet_emotions/{model_name}.pt')
    model.eval()
    torch_out=model(dummy_input)

    onnx_file=f'../models/affectnet_emotions/onnx/{model_name}.onnx'
    torch.onnx.export(model, dummy_input, onnx_file, opset_version=11,do_constant_folding=True,export_params=True,input_names=['input'],output_names=['output'],dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})

    onnx_model=onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    ort_session =ort.InferenceSession(onnx_file,providers=['CPUExecutionProvider'])

    x=np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    outputs = ort_session.run(None,{"input": x})
    print(outputs[0])

    print(model(torch.from_numpy(x)))

if __name__ == '__main__':
    for model_name in ['enet_b0_8_best_vgaf', 'enet_b0_8_best_afew', 'enet_b2_8', 'enet_b0_8_va_mtl', 'enet_b2_7']:
        img_size=260 if '_b2_' in model_name else 224
        convert_pt_to_onnx(model_name, img_size)