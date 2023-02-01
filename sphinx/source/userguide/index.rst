################
Installation
################

.. contents:: Table of Contents


*********************
Package Requirements
*********************

.. literalinclude:: ../../../requirements/requirements.in

****************
CI Requirements
****************

.. literalinclude:: ../../../requirements/requirements_ci.in

*************
Installation
*************

From pypi server

.. code-block:: bash

    pip install py4ai-analytics

From source

.. code-block:: bash

    git clone https://github.com/NicolaDonelli/py4ai-analytics
    cd "$(basename "https://github.com/NicolaDonelli/py4ai-analytics" .git)"
    make install
