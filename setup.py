from setuptools import setup, Extension

module = Extension(
    'cartpole',
    sources=['cartpole_module.c', 'cartpole.c'],
    include_dirs=['.'],
    extra_compile_args=['-std=c99'],
)

setup(
    name='cartpole',
    version='1.1',
    description='CartPole environment implemented in C',
    ext_modules=[module],
)