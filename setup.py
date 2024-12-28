#!/usr/bin/env python

from distutils.core import setup

setup(
    name='peft_sam',
    version='0.0.1',
    description='Parameter efficient finetuning methods for biomedical image segmentation using Segment Anything Models.',  # noqa
    author='Carolin Teuber, Anwai Archit, Constantin Pape',
    author_email='carolin.teuber@stud.uni-goettingen.de, anwai.archit@uni-goettingen.de, constantin.pape@informatik.uni-goettingen.de',  # noqa
    url='https://user.informatik.uni-goettingen.de/~pape41/',
    packages=['peft_sam'],
)
