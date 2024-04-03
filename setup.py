from setuptools import setup, find_packages

setup(
    name='embeddings_explorer',
    version='0.1.1',
    author='Edoardo Lanzini',
    author_email='edoardo.lanzini@gmail.com',
    description='A package for generating, exploring, and visualizing word embeddings graphs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/elanzini/pineapple_on_pizza',
    packages=find_packages(),
    install_requires=[
        'nltk>=3.8.0,<4.0',
        'numpy>=1.26.0,<2.0',
        'scikit-learn>=1.2.0,<2.0',
        'matplotlib>=3.7.0,<4.0',
        'transformers>=4.38.0,<5.0',
        'sentence-transformers>=2.6.0,<3.0',
        'networkx>=3.0,<4.0',
        'torch>=2.2.0,<3.0',
        'openai>=1.14.0,<2.0',
        'voyageai>=0.2.0,<1.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
