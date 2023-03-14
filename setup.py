from setuptools import setup

setup(
    name='asfarapi',
    version='0.1.0',
    description='Provides a simple way to train PyTorch models',
    url='https://github.com/asarfa/PyTorchModels',
    author='Sarfati Alban',
    author_email='alban.sarfati@studen-cs.fr',
    license='MIT',
    packages=['asfarapi'],
    install_requires=['numpy>=1.24.2',
                      'torch>=1.13.1',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
