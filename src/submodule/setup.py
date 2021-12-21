def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info

    info = get_info("npymath")
    config = Configuration("submodule", parent_package, top_path)
    config.add_extension(
        "submodule_c_name",
        ["example.cpp"],
        extra_info=info,
        extra_compile_args=["-O3", "-std=c++17", "-Wcpp", "-Wall"],
        extra_link_args=[],
        include_dirs=[],
        library_dirs=[],
    )
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
