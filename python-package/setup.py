from setuptools import setup, find_packages

requirements = [
    'numpy',
    'pillow',
    'torch',
    'torchvision',
    'timm'
]
setup(
    name='hsemotions',
    version='0.2',
    license='Apache-2.0',
    author="Andrey Savchenko",
    author_email='andrey.v.savchenko@gmail.com',
    packages=find_packages('.'),
    url='https://github.com/HSE-asavchenko/face-emotion-recognition',
    description='HSEmotions Python Library for Facial Emotion Recognition',
    keywords='face emotion recognition',
    install_requires=requirements,
)