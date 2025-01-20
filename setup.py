#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("peft_sam/__version__.py")["__version__"]


setup(
    name='peft_sam',
    description='Parameter Efficient Fine-Tuning (PEFT) methods for Segment Anything Models.',
    version=__version__,
    author='Carolin Teuber, Anwai Archit',
    author_email='carolin.teuber@stud.uni-goettingen.de, anwai.archit@uni-goettingen.de',
    url='https://github.com/computational-cell-analytics/PEFT_SAM',
    packages=['peft_sam'],
)
