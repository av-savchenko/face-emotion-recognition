from setuptools import setup, find_packages

requirements = [
    'numpy',
    'opencv-python',
    'onnx',
    'onnxruntime'
]
setup(
    name='hsemotion_onnx',
    version='0.2.1',
    license='Apache-2.0',
    author="Andrey Savchenko",
    author_email='andrey.v.savchenko@gmail.com',
    packages=find_packages('.'),
    download_url = 'https://github.com/HSE-asavchenko/face-emotion-recognition/archive/v0.2.1.tar.gz',
    url='https://github.com/HSE-asavchenko/face-emotion-recognition',
    description='HSEmotionONNX Python Library for Facial Emotion Recognition',
    keywords='face emotion recognition',
    install_requires=requirements,
)
