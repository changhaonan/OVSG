from setuptools import setup

setup(
   name='ovsg',
   version='1.0',
   description='A useful module',
   author='Haonan Chang',
   author_email='chnme40cs@gmail.com',
   packages=['ovsg'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)