from setuptools import setup, find_packages

exec(open('./version.py').read())

setup(
  name = 'inai-videogen-pixart',
  package_dir = {
    'data': 'data',
    'model': 'model',
    'pipeline': 'pipeline',
  },
  version = __version__,
  description = 'videogen from INAI - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'liruoyu01',
  author_email = 'liruoyu@in.ai',
  url = 'https://github.com/liruoyu01/inai_videogen_pixart',
  keywords = [
    'genai',
    'transformer',
    'generative video model'
  ],
  install_requires=[
    'accelerate>=0.24.0',
    'einops>=0.7.0',
    'ema-pytorch>=0.2.4',
    'pytorch-warmup',
    'opencv-python',
    'pillow',
    'numpy',
    'torch',
    'torchvision',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.6',
  ],
)
