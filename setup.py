#!/usr/bin/env python3

from pathlib import Path

from numpy.distutils.core import setup
import setuptools

project_dir = Path(__file__).parent

exec(open('version.txt').read())

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('src')
    return config


if __name__ == "__main__":
    setup(
        name="mxifpublic",
        version=__version__,
        description="",
        long_description=project_dir.joinpath("README.rst").read_text(encoding="utf-8"),
        keywords=["python"],
        author="BostonGene BRI",
        author_email="",
        url="",
        package_dir={"": "src"},
        python_requires=">=3.8",

        include_package_data=True,
        package_data={
            "submodule": ["py.typed"],
            "cell_typing": ["py.typed"],
            "permutation": ["py.typed"],
            "plotting": ["py.typed"],
            "community": ["py.typed"],
        },

        setup_requires=['wheel'],
        install_requires=project_dir.joinpath("requirements.txt").read_text().split("\n"),
        configuration=configuration,
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
    )
