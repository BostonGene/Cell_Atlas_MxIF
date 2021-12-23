from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("mxifpublic", parent_package, top_path)
    config.add_subpackage("submodule")
    return config


if __name__ == "__main__":
    print("This is the wrong setup.py file to run")