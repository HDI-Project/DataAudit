#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'coverage>=4.5.1',
    'pytest>=3.4.2',
    'tox>=2.9.1'
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="DataAudit",
    extras_require={
        'test': tests_require
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='dataaudit',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='data-audit',
    packages=find_packages(include=['dataaudit', 'dataaudit.*']),
    python_requires='>=3.4',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/DataAudit',
    version='0.1.0-dev',
    zip_safe=False,
)
