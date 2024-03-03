from setuptools import find_packages, setup

setup(
    name='t_ragx',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.0.3',
    description='Enhancing translation with RAG-powered large language models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ray Liu',
    email='ray@rayliu.ca',
    license='MIT',
)
