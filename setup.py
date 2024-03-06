from setuptools import find_packages, setup

setup(
    name='t_ragx',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.0.10',
    url="https://github.com/rayliuca/T-Ragx",
    description='Enhancing translation with RAG-powered large language models',
    long_description=open('README.md', encoding="utf8").read(),
    long_description_content_type='text/markdown',
    author='Ray Liu',
    email='ray@rayliu.ca',
    license='MIT',
    project_urls={
        "Bug Reports": "https://github.com/rayliuca/T-Ragx/issues",
        "Source": "https://github.com/rayliuca/T-Ragx",
    },

    install_requires=[
        'transformers>=4.38',
        'elasticsearch',
        'OpenCC',
        'levenshtein',
        'unbabel-comet>=2.2.1',
    ],
    keywords=[
        'machine-translation',
        'computer-aided-translation',
        'computer-assisted-translation',
        'rag',
        'retrieval-augmented-generation',
        'large-language-models',
        'llm',
    ]
)
