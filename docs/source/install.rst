Installation
============

General Remarks
---------------

- There are two different variations of TensorFlow that you might wish to install, depending on whether you would like TensorFlow to run on your CPU or GPU, namely :ref:`tensorflow_cpu` and :ref:`tensorflow_gpu`. I will proceed to document both and you can choose which one you wish to install.

- If you wish to install both TensorFlow variants on your machine, ideally you should install each variant under a different (virtual) environment. If you attempt to install both :ref:`tensorflow_cpu` and :ref:`tensorflow_gpu`, without making use of virtual environments, you will either end up failing, or when we later start running code there will always be an uncertainty as to which variant is being used to execute your code.

- To ensure that we have no package conflicts and/or that we can install several different versions/variants of TensorFlow (e.g. CPU and GPU), it is generally recommended to use a virtual environment of some sort. For the purposes of this tutorial we will be creating and managing our virtual environments using Anaconda, but you are welcome to use the virtual environment manager of your choice (e.g. virtualenv). 

Install Anaconda Python 3.7 (Optional)
--------------------------------------
Although having Anaconda is not a requirement in order to install and use TensorFlow, I suggest doing so, due to it's intuitive way of managing packages and setting up new virtual environments. Anaconda is a pretty useful tool, not only for working with TensorFlow, but in general for anyone working in Python, so if you haven't had a chance to work with it, now is a good chance.

.. tabs::

    .. tab:: Windows

        - Go to `<https://www.anaconda.com/download/>`_
        - Download `Anaconda Python 3.7 version for Windows <https://www.anaconda.com/distribution/#windows>`_
        - Run the downloaded executable (``.exe``) file to begin the installation. See `here <https://docs.anaconda.com/anaconda/install/windows/>`_ for more details.
        - (Optional) In the next step, check the box "Add Anaconda to my PATH environment variable". This will make Anaconda your default Python distribution, which should ensure that you have the same default Python distribution across all editors.

    .. tab:: Linux

        - Go to `<https://www.anaconda.com/download/>`_
        - Download `Anaconda Python 3.7 version for Linux <https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh>`_
        - Run the downloaded bash script (``.sh``) file to begin the installation. See `here <https://docs.anaconda.com/anaconda/install/linux/>`_ for more details.
        - When prompted with the question "Do you wish the installer to prepend the Anaconda<2 or 3> install location to PATH in your /home/<user>/.bashrc ?", answer "Yes". If you enter "No", you must manually add the path to Anaconda or conda will not work.

.. _tf_install:

TensorFlow Installation 
-----------------------

As mentioned in the Remarks section, there exist two generic variants of TensorFlow, which utilise different hardware on your computer to run their computationally heavy Machine Learning algorithms.
    
    1. The simplest to install, but also in most cases the slowest in terms of performance, is :ref:`tensorflow_cpu`, which runs directly on the CPU of your machine. 
    2. Alternatively, if you own a (compatible) Nvidia graphics card, you can take advantage of the available CUDA cores to speed up the computations performed by TensorFlow, in which case you should follow the guidelines for installing :ref:`tensorflow_gpu`.  

.. _tensorflow_cpu:

TensorFlow CPU
~~~~~~~~~~~~~~

Getting setup with an installation of TensorFlow CPU can be done in 3 simple steps.

.. important:: The term `Terminal` will be used to refer to the Terminal of your choice (e.g. Command Prompt, Powershell, etc.)

Create a new Conda virtual environment (Optional)
*************************************************
* Open a new `Terminal` window
* Type the following command:

    .. code-block:: posh

        conda create -n tensorflow_cpu pip python=3.7

* The above will create a new virtual environment with name ``tensorflow_cpu``
* Now lets activate the newly created virtual environment by running the following in the `Terminal` window:

    .. code-block:: posh

        activate tensorflow_cpu

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

.. code-block:: ps1con

    (tensorflow_cpu) C:\Users\sglvladi>

Install TensorFlow CPU for Python
*********************************
- Open a new `Terminal` window and activate the `tensorflow_cpu` environment (if you have not done so already)
- Once open, type the following on the command line:

    .. code-block:: posh

        pip install --ignore-installed --upgrade tensorflow==1.14

- Wait for the installation to finish

Test your Installation
**********************
- Open a new `Terminal` window and activate the `tensorflow_cpu` environment (if you have not done so already)
- Start a new Python interpreter session by running:

    .. code-block:: posh

        python

- Once the interpreter opens up, type:

    .. code-block:: python

        >>> import tensorflow as tf

- If the above code shows an error, then check to make sure you have activated the `tensorflow_cpu` environment and that tensorflow_cpu was successfully installed within it in the previous step.
- Then run the following:

    .. code-block:: python

        >>> hello = tf.constant('Hello, TensorFlow!')
        >>> sess = tf.Session()

- Once the above is run, if you see a print-out similar (or identical) to the one below, it means that you could benefit from installing TensorFlow by building the sources that correspond to you specific CPU. Everything should still run as normal, but potentially slower than if you had built TensorFlow from source.

    .. code-block:: python

        2019-02-28 11:59:25.810663: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

- Finally, run the following:

    .. code-block:: python

        >>> print(sess.run(hello))
        b'Hello, TensorFlow!'

.. _tensorflow_gpu:

TensorFlow GPU
~~~~~~~~~~~~~~

The installation of `TensorFlow GPU` is slightly more involved than that of `TensorFlow CPU`, mainly due to the need of installing the relevant Graphics and CUDE drivers. There's a nice Youtube tutorial (see `here <https://www.youtube.com/watch?v=RplXYjxgZbw>`_), explaining how to install TensorFlow GPU. Although it describes different versions of the relevant components (including TensorFlow itself), the installation steps are generally the same with this tutorial. 

Before proceeding to install TensorFlow GPU, you need to make sure that your system can satisfy the following requirements:

+-------------------------------------+
| Prerequisites                       |
+=====================================+
| Nvidia GPU (GTX 650 or newer)       |
+-------------------------------------+
| CUDA Toolkit v10.0                  |
+-------------------------------------+
| CuDNN 7.6.5                         |
+-------------------------------------+ 
| Anaconda with Python 3.7 (Optional) |
+-------------------------------------+

.. _cuda_install:

Install CUDA Toolkit
***********************
.. tabs::

    .. tab:: Windows

        Follow this `link <https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork>`_ to download and install CUDA Toolkit 10.0.

    .. tab:: Linux

        Follow this `link <https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64>`_ to download and install CUDA Toolkit 10.0 for your Linux distribution.

.. _cudnn_install:

Install CUDNN
****************
.. tabs::

    .. tab:: Windows

        - Go to `<https://developer.nvidia.com/rdp/cudnn-download>`_
        - Create a user profile if needed and log in
        - Select `cuDNN v7.6.5 (Nov 5, 2019), for CUDA 10.0 <https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-10>`_
        - Download `cuDNN v7.6.5 Library for Windows 10 <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-windows10-x64-v7.6.5.32.zip>`_
        - Extract the contents of the zip file (i.e. the folder named ``cuda``) inside ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\``, where ``<INSTALL_PATH>`` points to the installation directory specified during the installation of the CUDA Toolkit. By default ``<INSTALL_PATH>`` = ``C:\Program Files``.

    .. tab:: Linux

        - Go to `<https://developer.nvidia.com/rdp/cudnn-download>`_
        - Create a user profile if needed and log in
        - Select `cuDNN v7.6.5 (Nov 5, 2019), for CUDA 10.0  <https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-10>`_
        - Download `cuDNN v7.6.5 Library for Linux <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz>`_
        - Follow the instructions under Section 2.3.1 of the `CuDNN Installation Guide <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux>`_ to install CuDNN.

.. _set_env:

Environment Setup
*****************
.. tabs::

    .. tab:: Windows

        - Go to `Start` and Search "environment variables"
        - Click "Edit the system environment variables". This should open the "System Properties" window
        - In the opened window, click the "Environment Variables..." button to open the "Environment Variables" window.
        - Under "System variables", search for and click on the ``Path`` system variable, then click "Edit..."
        - Add the following paths, then click "OK" to save the changes:
            
            - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin``
            - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp``
            - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64``
            - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\cuda\bin``

    .. tab:: Linux 

        As per Section 7.1.1 of the `CUDA Installation Guide for Linux <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux>`_, append the following lines to ``~/.bashrc``:

        .. code-block:: bash

            # CUDA related exports
            export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
            export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

Update your GPU drivers (Optional)
**********************************
If during the installation of the CUDA Toolkit (see :ref:`cuda_install`) you selected the `Express Installation` option, then your GPU drivers will have been overwritten by those that come bundled with the CUDA toolkit. These drivers are typically NOT the latest drivers and, thus, you may wish to updte your drivers.

- Go to `<http://www.nvidia.com/Download/index.aspx>`_
- Select your GPU version to download
- Install the driver for your chosen OS

Create a new Conda virtual environment
**************************************
* Open a new `Terminal` window
* Type the following command:

    .. code-block:: posh

        conda create -n tensorflow_gpu pip python=3.7

* The above will create a new virtual environment with name ``tensorflow_gpu``
* Now lets activate the newly created virtual environment by running the following in the `Anaconda Promt` window:

    .. code-block:: posh

        activate tensorflow_gpu

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

.. code-block:: ps1con

    (tensorflow_gpu) C:\Users\sglvladi>

Install TensorFlow GPU for Python
*********************************
- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- Once open, type the following on the command line:

    .. code-block:: posh

        pip install --upgrade tensorflow-gpu==1.14

- Wait for the installation to finish

Test your Installation
**********************
- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- Start a new Python interpreter session by running:

    .. code-block:: posh

        python

- Once the interpreter opens up, type:

    .. code-block:: python

        >>> import tensorflow as tf

- If the above code shows an error, then check to make sure you have activated the `tensorflow_gpu` environment and that tensorflow_gpu was successfully installed within it in the previous step.
- Then run the following:

    .. code-block:: python

        >>> hello = tf.constant('Hello, TensorFlow!')
        >>> sess = tf.Session()
- Once the above is run, you should see a print-out similar (but not identical) to the one bellow:

    .. code-block:: python

        2019-11-25 07:20:32.415386: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
        2019-11-25 07:20:32.449116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
        name: GeForce GTX 1070 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
        pciBusID: 0000:01:00.0
        2019-11-25 07:20:32.455223: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
        2019-11-25 07:20:32.460799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
        2019-11-25 07:20:32.464391: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
        2019-11-25 07:20:32.472682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
        name: GeForce GTX 1070 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
        pciBusID: 0000:01:00.0
        2019-11-25 07:20:32.478942: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
        2019-11-25 07:20:32.483948: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
        2019-11-25 07:20:33.181565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
        2019-11-25 07:20:33.185974: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
        2019-11-25 07:20:33.189041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
        2019-11-25 07:20:33.193290: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6358 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)

- Finally, run the following:

    .. code-block:: python

        >>> print(sess.run(hello))
        b'Hello, TensorFlow!'

.. _tf_models_install:

TensorFlow Models Installation 
------------------------------

Now that you have installed TensorFlow, it is time to install the models used by TensorFlow to do its magic.

Install Prerequisites
~~~~~~~~~~~~~~~~~~~~~

Building on the assumption that you have just created your new virtual environment (whether that's `tensorflow_cpu`, `tensorflow_gpu` or whatever other name you might have used), there are some packages which need to be installed before installing the models.

+---------------------------------------------+
| Prerequisite packages                       |
+--------------+------------------------------+
| Name         | Tutorial version-build       |
+==============+==============================+
| pillow       | 6.2.1-py37hdc69c19_0         |
+--------------+------------------------------+
| lxml         | 4.4.1-py37h1350720_0         |
+--------------+------------------------------+
| jupyter      | 1.0.0-py37_7                 |
+--------------+------------------------------+
| matplotlib   | 3.1.1-py37hc8f65d3_0         |
+--------------+------------------------------+
| opencv       | 3.4.2-py37hc319ecb_0         |
+--------------+------------------------------+
| pathlib      | 1.0.1-cp37                   |
+--------------+------------------------------+

The packages can be installed using ``conda`` by running:

.. code-block:: posh

    conda install <package_name>(=<version>), <package_name>(=<version>), ..., <package_name>(=<version>)

where ``<package_name>`` can be replaced with the name of the package, and optionally the package version can be specified by adding the optional specifier ``=<version>`` after ``<package_name>``. For example, to simply install all packages at their latest versions you can run:

.. code-block:: posh

    conda install pillow, lxml, jupyter, matplotlib, opencv, cython

Alternatively, if you don't want to use Anaconda you can install the packages using ``pip``:

.. code-block:: posh

    pip install <package_name>(==<version>) <package_name>(==<version>) ... <package_name>(==<version>)

but you will need to install ``opencv-python`` instead of ``opencv``.

Downloading the TensorFlow Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: To ensure compatibility with the chosen version of Tensorflow (i.e. ``1.14.0``), it is generally recommended to use one of the `Tensorflow Models releases <https://github.com/tensorflow/models/releases>`_, as they are most likely to be stable. Release ``v1.13.0`` is the last unofficial release before ``v2.0`` and therefore is the one used here.

- Create a new folder under a path of your choice and name it ``TensorFlow``. (e.g. ``C:\Users\sglvladi\Documents\TensorFlow``).
- From your `Terminal` ``cd`` into the ``TensorFlow`` directory.
- To download the models you can either use `Git <https://git-scm.com/downloads>`_ to clone the `TensorFlow Models v.1.13.0 release <https://github.com/tensorflow/models/tree/r1.13.0>`_ inside the ``TensorFlow`` folder, or you can simply download it as a `ZIP <https://github.com/tensorflow/models/archive/r1.13.0.zip>`_ and extract it's contents inside the ``TensorFlow`` folder. To keep things consistent, in the latter case you will have to rename the extracted folder ``models-r1.13.0`` to ``models``.
- You should now have a single folder named ``models`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

.. code-block:: bash

    TensorFlow
    └─ models
        ├── official
        ├── research
        ├── samples
        └── tutorials

Protobuf Installation/Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be downloaded and compiled. 

This should be done as follows:

- Head to the `protoc releases page <https://github.com/google/protobuf/releases>`_
- Download the latest ``protoc-*-*.zip`` release (e.g. ``protoc-3.11.0-win64.zip`` for 64-bit Windows)
- Extract the contents of the downloaded ``protoc-*-*.zip`` in a directory ``<PATH_TO_PB>`` of your choice (e.g. ``C:\Program Files\Google Protobuf``)
- Extract the contents of the downloaded ``protoc-*-*.zip``, inside ``C:\Program Files\Google Protobuf``
- Add ``<PATH_TO_PB>`` to your ``Path`` environment variable (see :ref:`set_env`)
- In a new `Terminal` [#]_, ``cd`` into ``TensorFlow/models/research/`` directory and run the following command:

    .. code-block:: python

        # From within TensorFlow/models/research/
        protoc object_detection/protos/*.proto --python_out=.

.. important::

    If you are on Windows and using Protobuf 3.5 or later, the multi-file selection wildcard (i.e ``*.proto``) may not work but you can do one of the following:

    .. tabs::

        .. tab:: Windows Powershell

            .. code-block:: python

                # From within TensorFlow/models/research/
                Get-ChildItem object_detection/protos/*.proto | foreach {protoc "object_detection/protos/$($_.Name)" --python_out=.}


        .. tab:: Command Prompt

            .. code-block:: python

                    # From within TensorFlow/models/research/
                    for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.


.. [#] NOTE: You MUST open a new `Terminal` for the changes in the environment variables to take effect.


Adding necessary Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the ``Tensorflow\models\research\object_detection`` package by running the following from ``Tensorflow\models\research``:

    .. code-block:: python

        # From within TensorFlow/models/research/
        pip install .

2. Add `research/slim` to your ``PYTHONPATH``:

.. tabs::

    .. tab:: Windows

        - Go to `Start` and Search "environment variables"
        - Click "Edit the system environment variables". This should open the "System Properties" window
        - In the opened window, click the "Environment Variables..." button to open the "Environment Variables" window.
        - Under "System variables", search for and click on the ``PYTHONPATH`` system variable,

            - If it exists then click "Edit..." and add ``<PATH_TO_TF>\TensorFlow\models\research\slim`` to the list
            - If it doesn't already exist, then click "New...", under "Variable name" type ``PYTHONPATH`` and under "Variable value" enter ``<PATH_TO_TF>\TensorFlow\models\research\slim``

        - Then click "OK" to save the changes:

    .. tab:: Linux
    
        The `Installation docs <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>`_ suggest that you either run, or add to ``~/.bashrc`` file, the following command, which adds these packages to your PYTHONPATH:

        .. code-block:: bash

            # From within tensorflow/models/research/
            export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim

    where, in both cases, ``<PATH_TO_TF>`` replaces the absolute path to your ``TensorFlow`` folder. (e.g. ``<PATH_TO_TF>`` = ``C:\Users\sglvladi\Documents`` if ``TensorFlow`` resides within your ``Documents`` folder)

.. _tf_models_install_coco:

COCO API installation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``pycocotools`` package should be installed if you are interested in using COCO evaluation metrics, as discussed in :ref:`evaluation_sec`.

.. tabs::

    .. tab:: Windows

        Run the following command to install ``pycocotools`` with Windows support:

        .. code-block:: bash

            pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


        Note that, according to the `package's instructions <https://github.com/philferriere/cocoapi#this-clones-readme>`_, Visual C++ 2015 build tools must be installed and on your path. If they are not, make sure to install them from `here <https://go.microsoft.com/fwlink/?LinkId=691126>`_.

    .. tab:: Linux
    
        Download `cocoapi <https://github.com/cocodataset/cocoapi>`_ to a directory of your choice, then ``make`` and copy the pycocotools subfolder to the ``Tensorflow/models/research`` directory, as such: 

        .. code-block:: bash

            git clone https://github.com/cocodataset/cocoapi.git
            cd cocoapi/PythonAPI
            make
            cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/

.. note:: The default metrics are based on those used in Pascal VOC evaluation.

    - To use the COCO object detection metrics add ``metrics_set: "coco_detection_metrics"`` to the ``eval_config`` message in the config file.

    - To use the COCO instance segmentation metrics add ``metrics_set: "coco_mask_metrics"`` to the ``eval_config`` message in the config file.


.. _test_tf_models:

Test your Installation
~~~~~~~~~~~~~~~~~~~~~~

- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- ``cd`` into ``TensorFlow\models\research\object_detection`` and run the following command:

    .. code-block:: posh

        # From within TensorFlow/models/research/object_detection
        jupyter notebook

- This should start a new ``jupyter notebook`` server on your machine and you should be redirected to a new tab of your default browser.

- Once there, simply follow `sentdex's Youtube video <https://youtu.be/COlbP62-B-U?t=7m23s>`_ to ensure that everything is running smoothly.

- When done, your notebook should look similar to the image bellow:

    .. image:: ./_static/object_detection_tutorial_output.png
       :width: 90%
       :alt: alternate text
       :align: center

.. important::
    1. If no errors appear, but also no images are shown in the notebook, try adding ``%matplotlib inline`` at the start of the last cell, as shown by the highlighted text in the image bellow:

    .. image:: ./_static/object_detection_tutorial_err.png
       :width: 90%
       :alt: alternate text
       :align: center


    2. If Python crashes when running the last cell, have a look at the `Terminal` window you used to run ``jupyter notebook`` and check for an error similar (maybe identical) to the one below:

        .. code-block:: python

            2018-03-22 03:07:54.623130: E C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\cuda\cuda_dnn.cc:378] Loaded runtime CuDNN library: 7101 (compatibility version 7100) but source was compiled with 7003 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.

        - If the above line is present in the printed debugging, it means that you have not installed the correct version of the cuDNN libraries. In this case make sure you re-do the :ref:`cudnn_install` step, making sure you instal cuDNN v7.6.5.


.. n

.. _labelImg_install:

LabelImg Installation
---------------------

There exist several ways to install ``labelImg``. Below are 3 of the most common.

Get from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
2. Run the following command to install ``labelImg``:

.. code-block:: bash

    pip install labelImg

3. ``labelImg`` can then be run as follows:

.. code-block:: bash

    labelImg
    # or
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

Use precompiled binaries (Easy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Precompiled binaries for both Windows and Linux can be found `here <http://tzutalin.github.io/labelImg/>`_ .

Installation is the done in three simple steps:

1. Inside you ``TensorFlow`` folder, create a new directory, name it ``addons`` and then ``cd`` into it.

2. Download the latest binary for your OS from `here <http://tzutalin.github.io/labelImg/>`_. and extract its contents under ``Tensorflow/addons/labelImg``.

3. You should now have a single folder named ``addons\labelImg`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

.. code-block:: bash

    TensorFlow
    ├─ addons
    │   └── labelImg
    └─ models
        ├── official
        ├── research
        ├── samples
        └── tutorials

4. ``labelImg`` can then be run as follows:

.. code-block:: bash

    # From within Tensorflow/addons/labelImg
    labelImg
    # or
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

Build from source (Hard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The steps for installing from source follow below.

**1. Download labelImg**

- Inside you ``TensorFlow`` folder, create a new directory, name it ``addons`` and then ``cd`` into it.
- To download the package you can either use `Git <https://git-scm.com/downloads>`_ to clone the `labelImg repo <https://github.com/tzutalin/labelImg>`_ inside the ``TensorFlow\addons`` folder, or you can simply download it as a `ZIP <https://github.com/tzutalin/labelImg/archive/master.zip>`_ and extract it's contents inside the ``TensorFlow\addons`` folder. To keep things consistent, in the latter case you will have to rename the extracted folder ``labelImg-master`` to ``labelImg``. [#]_
- You should now have a single folder named ``addons\labelImg`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

.. code-block:: bash

    TensorFlow
    ├─ addons
    │   └── labelImg
    └─ models
        ├── official
        ├── research
        ├── samples
        └── tutorials

.. [#] The latest repo commit when writing this tutorial is `8d1bd68 <https://github.com/tzutalin/labelImg/commit/8d1bd68ab66e8c311f2f45154729bba301a81f0b>`_.

**2. Install dependencies and compiling package**

- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- ``cd`` into ``TensorFlow\addons\labelImg`` and run the following commands:

    .. tabs:: 

        .. tab:: Windows

            .. code-block:: bash
                
                conda install pyqt=5
                pyrcc5 -o libs/resources.py resources.qrc
            
        .. tab:: Linux 

            .. code-block:: bash

                sudo apt-get install pyqt5-dev-tools
                sudo pip install -r requirements/requirements-linux-python3.txt
                make qt5py3


**3. Test your installation**

- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- ``cd`` into ``TensorFlow\addons\labelImg`` and run the following command:

    .. code-block:: posh

        # From within Tensorflow/addons/labelImg
        python labelImg.py
        # or       
        python  labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]



