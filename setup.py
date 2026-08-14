from setuptools import setup, find_packages

setup(
    name='apa',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['torch>=2.0.0'],
    extras_require={
        'examples': [
            'torchvision>=0.15.0',
            'tqdm',
        ],
    },
    description='Adaptive Precision Architecture — Reckless-start FP8 training with dynamic precision escalation for PyTorch',
    author='APA Research',
    license='MIT',
)
