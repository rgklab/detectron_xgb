import setuptools

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'detectron_xgb'))

setuptools.setup(name='detectron_xgb',
                 version='0.0.1',
                 url='https://github.com/rgklab/detectron_xgb',
                 description='Robust and Highly Sensitive Covariate Shift Detection using XGBoost',
                 author='Tom Ginsberg',
                 author_email='tomginsberg@cs.toronto.edu',
                 python_requires='>=3.10',
                 install_requires=[
                     'xgboost>=1.7.3',
                     'tqdm>=4.64.1',
                     'numpy>=1.24.2',
                     'pandas>=1.5.3',
                     'scikit-learn>=1.2.1'
                 ],
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Intended Audience :: Science/Research',
                     "Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                 ]
                 )
