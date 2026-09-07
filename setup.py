import os
import sys
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

# Detect CUDA environment for building native C++/CUDA extension
build_cuda = os.environ.get('APA_BUILD_CUDA', '1') == '1'

if build_cuda:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
        
        csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
        sources = [
            os.path.join(csrc_dir, 'apa_cuda.cpp'),
            os.path.join(csrc_dir, 'fp8_quantize.cu'),
        ]
        
        all_exist = all(os.path.isfile(s) for s in sources)
        
        if all_exist and CUDA_HOME is not None and torch.cuda.is_available():
            nvcc_flags = ['-O3', '--use_fast_math', '-std=c++17']
            
            # Architecture targets: Blackwell (sm_120), Hopper (sm_90), Ada Lovelace (sm_89)
            nvcc_flags.extend([
                '-gencode=arch=compute_120,code=sm_120',
                '-gencode=arch=compute_90,code=sm_90',
                '-gencode=arch=compute_89,code=sm_89',
            ])
            
            # Detect current active GPU architecture if available
            try:
                major, minor = torch.cuda.get_device_capability(0)
                current_arch = f"{major}{minor}"
                current_flag = f"-gencode=arch=compute_{current_arch},code=sm_{current_arch}"
                if current_flag not in nvcc_flags:
                    nvcc_flags.append(current_flag)
            except Exception:
                pass

            cxx_flags = ['-O3', '-std=c++17']
            if sys.platform == 'win32':
                cxx_flags = ['/O2', '/std:c++17']

            ext_modules.append(
                CUDAExtension(
                    name='apa_cuda',
                    sources=sources,
                    extra_compile_args={
                        'cxx': cxx_flags,
                        'nvcc': nvcc_flags,
                    },
                )
            )
            cmdclass['build_ext'] = BuildExtension
            print("[APA Build] Native CUDA extension 'apa_cuda' configured with Blackwell sm_120 support.")
    except Exception as e:
        print(f"[APA Build] Notice: Skipping native C++/CUDA extension compilation ({e}). Pure Python fallback will be used.")

setup(
    name='apa',
    version='0.2.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.10',
    install_requires=['torch>=2.0.0'],
    extras_require={
        'examples': [
            'torchvision>=0.15.0',
            'tqdm',
            'datasets',
            'huggingface_hub',
            'pillow',
        ],
    },
    description='Adaptive Precision Ascension — Reckless-start FP8 training with dynamic precision escalation for PyTorch',
    author='APA Research',
    license='MIT',
)
