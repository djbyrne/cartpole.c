from setuptools import setup, Extension
import os

module = Extension(
    'cartpole',
    sources=['cartpole_module.c', 'cartpole.c'],
    include_dirs=[
        '.',  # Current directory
        '/opt/homebrew/opt/llvm/include',  # Include path for LLVM
    ],
    extra_compile_args=[
        '-std=c99',
        '-fopenmp',  # Enable OpenMP
    ],
    extra_link_args=[
        '-fopenmp',  # Link with OpenMP libraries
        '-L/opt/homebrew/opt/llvm/lib',  # Library path for LLVM
    ],
)

setup(
    name='cartpole',
    version='1.0',
    description='CartPole environment with OpenMP support',
    ext_modules=[module],
)