def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info

    info = get_info("npymath")
    config = Configuration("permutation", parent_package, top_path)
    config.add_extension(
        "permutation_c",
        ["permutation.cpp"],
        extra_info=info,
        extra_compile_args=["-fopenmp", "-Ofast", "-std=c++17", "-Wcpp", "-Wall"],
        extra_link_args=["-lgomp"],
        include_dirs=[],
        library_dirs=[],
    )
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
