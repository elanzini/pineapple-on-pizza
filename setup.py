from setuptools import setup, find_packages

setup(
    name='embeddings_explorer',
    version='0.1.0',
    author='Edoardo Lanzini',
    author_email='edoardo.lanzini@gmail.com',
    description='A package for generating, exploring, and visualizing word embeddings graphs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/elanzini/pineapple_on_pizza',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'networkx',
        'transformers',
        'sentence-transformers',
        'torch',
        'openai',
        'voyageai'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
