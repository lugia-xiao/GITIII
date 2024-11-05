from setuptools import setup, find_packages

setup(
    name='gitiii',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'gitiii': [
            'data/interactions_human.pth',
            'data/interactions_human_nonichenetv2.pth',
            'data/interactions_mouse.pth',
            'data/interactions_mouse_nonichenetv2.pth',
            'data/__init__.py'
        ]
    },
    install_requires=[
        'torch',
        'pandas',
        'monotonicnetworks',
        'scipy',
        'scanpy',
        'anndata',
        'statsmodels',
        'scikit-learn',  # scikit-learn is installed using 'scikit-learn' not 'sklearn'
        'numpy',
        'matplotlib',
        'maxfuse',
        'seaborn'
    ],
    python_requires='>=3.6',
    license='GPL-3.0',
    description='Investigation of pair-wise single cell interactions through statistically interpreting spatial cell state correlations learned by self-supervised graph inductive bias transformer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Xiao Xiao',
    author_email='xiao.xiao.xx244@yale.edu',
    url='https://github.com/lugia.xiao/gitiii',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
)
