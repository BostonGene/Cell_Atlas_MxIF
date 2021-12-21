MxIFPublic
================
Description

Requirements
------------

Python 3.7+.

Dependencies
------------

Dependencies are defined in:

- ``requirements.in``

- ``requirements.txt``

- ``dev-requirements.in``

- ``dev-requirements.txt``

Virtual Environments
^^^^^^^^^^^^^^^^^^^^

It is best practice during development to create an isolated
`Python virtual environment <https://docs.python.org/3/library/venv.html>`_ using the
``venv`` standard library module. This will keep dependant Python packages from interfering
with other Python projects on your system.

On \*Nix:

.. code-block:: bash

    # On Python 3.9+, add --upgrade-deps
    $ python3 -m venv venv
    $ source venv/bin/activate

Once activated, it is good practice to update core packaging tools (``pip``, ``setuptools``, and
``wheel``) to the latest versions.

.. code-block:: bash

    (venv) $ python -m pip install --upgrade pip setuptools wheel

(Applications Only) Locking Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project uses `pip-tools <https://github.com/jazzband/pip-tools>`_ to lock project
dependencies and create reproducible virtual environments.

**Note:** *Library* projects should not lock their ``requirements.txt``. Since ``python-blueprint``
also has a CLI application, this end-user application example is used to demonstrate how to
lock application dependencies.

To update dependencies:

.. code-block:: bash

    (venv) $ python -m pip install pip-tools
    (venv) $ pip-compile --upgrade
    (venv) $ pip-compile --upgrade dev-requirements.in

After upgrading dependencies, run the unit tests as described in the `Unit Testing`_ section
to ensure that none of the updated packages caused incompatibilities in the current project.

Syncing Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To cleanly install your dependencies into your virtual environment:

.. code-block:: bash

    (venv) $ pip-sync requirements.txt dev-requirements.txt

Building
--------

You can build a package for a specific python version using:

.. code-block:: bash

    # Build for pyhton3.7
    (venv) $ tox -e build_wheel-py37

    # Build for pyhton3.8
    (venv) $ tox -e build_wheel-py37

    # Build for pyhton3.9
    (venv) $ tox -e build_wheel-py37

Code Style Checking
^^^^^^^^^^^^^^^^^^^

`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ is the universally accepted style
guide for Python code. PEP8 code compliance is verified using `flake8 <http://flake8.pycqa.org/>`_.
flake8 is configured in the ``[flake8]`` section of ``tox.ini``. Extra flake8 plugins
are also included:

- ``pep8-naming``: Ensure functions, classes, and variables are named with correct casing.


Automated Code Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^

Code is automatically formatted using `black <https://github.com/psf/black>`_. Imports are
automatically sorted and grouped using `isort <https://github.com/PyCQA/isort/>`_.

These tools are configured by:

- ``pyproject.toml``

To automatically format code, run:

.. code-block:: bash

    (venv) $ tox -e black
    (venv) $ tox -e isort

To verify code has been formatted, such as in a CI job:

.. code-block:: bash

    (venv) $ tox -e black-check
    (venv) $ tox -e isort-check

Generated Documentation
^^^^^^^^^^^^^^^^^^^^^^^

Documentation that includes the ``README.rst`` and the Python project modules is automatically
generated using a `Sphinx <http://sphinx-doc.org/>`_ tox environment. Sphinx is a documentation
generation tool that is the defacto tool for Python documentation. Sphinx uses the
`RST <https://www.sphinx-doc.org/en/latest/usage/restructuredtext/basics.html>`_ markup language.

This project uses the
`napoleon <http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ plugin for
Sphinx, which renders Google-style and reST docstrings.  Yuo can find simple examples `here <http://queirozf.com/entries/python-docstrings-reference-examples>`_

.. code-block:: python

    """Computes the factorial through a recursive algorithm.

    Args:
        n: A positive input value.

    Raises:
        InvalidFactorialError: If n is less than 0.

    Returns:
        Computed factorial.
    """

The Sphinx project is configured in ``docs/conf.py``.

Build the docs using the ``docs`` tox environment (e.g. ``tox`` or ``tox -e docs``). Once built,
open ``docs/_build/index.html`` in a web browser.


Main website with `documentation <http://imaginomics2.devbg.us:45156/>`_ can be found on imaginomics2.devbg.us:45156.

At the moment, the documentation after pushing master brunch is automatically generated and updated to documentation site using `jenkins <https://www.jenkins.io/>`_.

On imaginomics2.devbg.us:45155 you can find local `jenkins server <http://imaginomics2.devbg.us:45155/>`_  for auto deploying.

Type Hinting
------------

`Type hinting <https://docs.python.org/3/library/typing.html>`_ allows developers to include
optional static typing information to Python source code. This allows static analyzers such
as `PyCharm <https://www.jetbrains.com/pycharm/>`_, `mypy <http://mypy-lang.org/>`_, or
`pytype <https://github.com/google/pytype>`_ to check that functions are used with the correct
types before runtime.

For
`PyCharm in particular <https://www.jetbrains.com/help/pycharm/type-hinting-in-product.html>`_,
the IDE is able to provide much richer auto-completion, refactoring, and type checking while
the user types, resulting in increased productivity and correctness.

This project uses the type hinting syntax introduced in Python 3:

.. code-block:: python

    def func(x: int) -> int:

Type checking is performed by mypy via ``tox -e type-check``. mypy is configured in ``setup.cfg``.

See also `awesome-python-typing <https://github.com/typeddjango/awesome-python-typing>`_.

Licensing
---------

You may also want to list the licenses of all of the packages that your Python project depends on.
To automatically list the licenses for all dependencies in ``requirements.txt`` (and their
transitive dependencies) using
`pip-licenses <https://github.com/raimon49/pip-licenses>`_:

.. code-block:: bash

    (venv) $ tox -e licenses
    ...
     Name        Version  License
     colorama  0.4.4    BSD License 
     numpy     1.20.1   BSD License