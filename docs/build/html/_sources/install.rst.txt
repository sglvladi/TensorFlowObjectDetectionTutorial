Installation
============

General Remarks
---------------

- There are two different variations of TensorFlow that you might wish to install, depending on whether you would like TensorFlow to run on your CPU or GPU, namely :ref:`tensorflow_cpu` and :ref:`tensorflow_gpu`. I will proceed to document both and you can choose which one you wish to install.

- If you wish to install both TensorFlow variants on your machine, ideally you should install each variant under a different (virtual) environment. If you attempt to install both :ref:`tensorflow_cpu` and :ref:`tensorflow_gpu`, without making use of virtual environments, you will either end up failing, or when we later start running code there will always be an uncertainty as to which variant is being used to execute your code.

- To ensure that we have no package conflicts and/or that we can install several different versions/variants of TensorFlow (e.g. CPU and GPU), it is generally recommended to use a virtual environment of some sort. For the purposes of this tutorial we will be creating and managing our virtual environments using Anaconda, but you are welcome to use the virtual environment manager of your choice (e.g. virtualenv). 

Install Anaconda Python 3.6 (Optional)
--------------------------------------
Although having Anaconda is not a requirement in order to install and use TensorFlow, I suggest doing so, due to it's intuitive way of managing packages and setting up new virtual environments. Anaconda is a pretty useful tool, not only for working with TensorFlow, but in general for anyone working in Python, so if you haven't had a chance to work with it, now is a good chance.

- Go to `<https://www.anaconda.com/download/>`_
- Download Anaconda Python 3.6 version
- If disk space is an issue for your machine, you could install the minified version of Anaconda (i.e. Miniconda).
- When prompted for a "Destination Folder" you can chose whichever you wish, but I generally tend to use ``C:\Anaconda3``, to keep things simple. Putting Anaconda under ``C:\Anaconda3`` also ensures that you don't get the awkward ```Destination Folder` contains spaces`` warning.

.. _tf_install:

TensorFlow Installation 
-----------------------

As mentioned in the Remarks section, there exist two generic variants of TensorFlow, which utilise different hardware on your computer to run their computationally heavy Machine Learning algorithms. The simplest to install, but also in most cases the slowest in terms of performance, is :ref:`tensorflow_cpu`, which runs directly on the CPU of your machine. Alternatively, if you own a (compatible) Nvidia graphics card, you can take advantage of the available CUDA cores to speed up the computations performed by TesnsorFlow, in which case you should follow the guidelines for installing :ref:`tensorflow_gpu`.  

.. _tensorflow_cpu:

TensorFlow CPU
~~~~~~~~~~~~~~

Getting setup with an installation of TensorFlow CPU can be done in 3 simple steps.

Create a new Conda virtual environment (Optional)
*************************************************
* Open a new `Anaconda/Command Prompt` window 
* Type the following command:

    .. code-block:: posh

        conda create -n tensorflow_cpu pip python=3.6

* The above will create a new virtual environment with name ``tensorflow_cpu``
* Now lets activate the newly created virtual environment by running the following in the `Anaconda Promt` window:

    .. code-block:: posh

        activate tensorflow_cpu

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

.. code-block:: ps1con

    (tensorflow_cpu) C:\Users\sglvladi>

Install TensorFlow CPU for Python
*********************************
- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_cpu` environment (if you have not done so already)
- Once open, type the following on the command line:

    .. code-block:: posh

        pip install --ignore-installed --upgrade tensorflow==1.7.0

- Wait for the installation to finish

Test your Installation
**********************
- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_cpu` environment (if you have not done so already)
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

- Once the above is run, if you see a print-out similar (but not identical) to the one below, it means that you could benefit from installing TensorFlow by building the sources that correspond to you specific CPU. Everything should still run as normal, just slower than if you had built TensorFlow from source.

    .. code-block:: python

        2018-03-21 22:10:18.682767: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

- Finally, for the sake of completing the test as described by TensorFlow themselves (see `here <https://www.tensorflow.org/install/install_windows#validate_your_installation>`_), let's run the following:

    .. code-block:: python

        >>> print(sess.run(hello))
        b'Hello, TensorFlow!'

.. _tensorflow_gpu:

TensorFlow GPU
~~~~~~~~~~~~~~

The installation of `TesnorFlow GPU` is slightly more involved than that of `TensorFlow CPU`, mainly due to the need of installing the relevant Graphics and CUDE drivers. There's a nice Youtube tutorial (see `here <https://www.youtube.com/watch?v=RplXYjxgZbw>`_), explaining how to install TensorFlow GPU. Although it describes different versions of the relevant components (including TensorFlow itself), the installation steps are generally the same with this tutorial. 

Before proceeding to install TesnsorFlow GPU, you need to make sure that your system can satisfy the following requirements:

+-------------------------------------+
| Prerequisites                       |
+=====================================+
| Nvidia GPU (GTX 650 or newer)       |
+-------------------------------------+
| CUDA Toolkit v9.0                   |
+-------------------------------------+
| CuDNN v7.0.5                        |
+-------------------------------------+ 
| Anaconda with Python 3.6 (Optional) |
+-------------------------------------+

.. _cuda_install:

Install CUDA Toolkit
***********************
Follow this `link <https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork>`_ to download and install CUDA Toolkit v9.0.

.. _cudnn_install:

Install CUDNN
****************
- Go to `<https://developer.nvidia.com/rdp/cudnn-download>`_
- Create a user profile if needed and log in
- Select `cuDNN v7.0.5 (Feb 28, 2018), for CUDA 9.0 <https://developer.nvidia.com/rdp/cudnn-download#a-collapse705-9>`_
- Download `cuDNN v7.0.5 Library for Windows 10 <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7>`_
- Extract the contents of the zip file (i.e. the folder named ``cuda``) inside ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\``, where ``<INSTALL_PATH>`` points to the installation directory specified during the installation of the CUDA Toolkit. By default ``<INSTALL_PATH>`` = ``C:\Program Files``.

.. _set_env:

Set Your Environment Variables
**********************************

- Go to `Start` and Search "environment variables"
- Click the Environment Variables button
- Click on the ``Path`` system variable and select edit
- Add the following paths:
    
    - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin``
    - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp``
    - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64``
    - ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\cuda\bin``

Update your GPU drivers (Optional)
**********************************
If during the installation of the CUDA Toolkit (see :ref:`cuda_install`) you selected the `Express Installation` option, then your GPU drivers will have been overwritten by those that come bundled with the CUDA toolkit. These drivers are typically NOT the latest drivers and, thus, you may wish to updte your drivers.

- Go to `<http://www.nvidia.com/Download/index.aspx>`_
- Select your GPU version to download
- Install the driver 

Create a new Conda virtual environment
**************************************
* Open a new `Anaconda/Command Prompt` window 
* Type the following command:

    .. code-block:: posh

        conda create -n tensorflow_gpu pip python=3.6

* The above will create a new virtual environment with name ``tensorflow_gpu``
* Now lets activate the newly created virtual environment by running the following in the `Anaconda Promt` window:

    .. code-block:: posh

        activate tensorflow_gpu

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

.. code-block:: ps1con

    (tensorflow_gpu) C:\Users\sglvladi>

Install TensorFlow GPU for Python
*********************************
- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- Once open, type the following on the command line:

    .. code-block:: posh

        pip install --ignore-installed --upgrade tensorflow-gpu==1.7.0

- Wait for the installation to finish

Test your Installation
**********************
- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_gpu` environment (if you have not done so already)
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

        2018-03-21 21:46:18.962971: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
        name: GeForce GTX 770 major: 3 minor: 0 memoryClockRate(GHz): 1.163
        pciBusID: 0000:02:00.0
        totalMemory: 2.00GiB freeMemory: 1.63GiB
        2018-03-21 21:46:18.978254: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
        2018-03-21 21:46:19.295152: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1414 MB memory) -> physical GPU (device: 0, name: GeForce GTX 770, pci bus id: 0000:02:00.0, compute capability: 3.0)

- Finally, for the sake of completing the test as described by TensorFlow themselves (see `here <https://www.tensorflow.org/install/install_windows#validate_your_installation>`_), let's run the following:

    .. code-block:: python

        >>> print(sess.run(hello))
        b'Hello, TensorFlow!'

.. _tf_models_install:

TensorFlow Models Installation 
------------------------------

Now that you have installed TensorFlow, it is time to install the models used by TensorFlow to do its magic.

Install Prerequisites
~~~~~~~~~~~~~~~~~~~~~

Building on the assumption that you have just created your new virtual environment (whether that's `tensorflow_cpu`,`tensorflow_gpu` or whatever other name you might have used), there are some packages which need to be installed before installing the models. 

+---------------------------------------------+
| Prerequisite packages                       |
+--------------+------------------------------+
| Name         | Tutorial version-build       |
+==============+==============================+
| pillow       | 5.0.0-py36h0738816_0         |
+--------------+------------------------------+
| lxml         | 4.2.0-py36heafd4d3_0         |
+--------------+------------------------------+
| jupyter      | 1.0.0-py36_4                 |
+--------------+------------------------------+
| matplotlib   | 2.2.2-py36h153e9ff_0         |
+--------------+------------------------------+
| opencv       | 3.3.1-py36h20b85fd_1         |
+--------------+------------------------------+

The packages can be install by running:

.. code-block:: posh

    conda install <package_name>(=<version>)

where ``<package_name>`` can be replaced with the name of the package, and optionally the package version can be specified by adding the optional specifier ``=<version>`` after ``<package_name>``. 

Alternatively, if you don't want to use Anaconda you can install the packages using ``pip``:

.. code-block:: posh

    pip install <package_name>(=<version>)

but you will need to install ``opencv-python`` instead of ``opencv``.

Downloading the TensorFlow Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a new folder under a path of your choice and name it ``TensorFlow``. (e.g. ``C:\Users\sglvladi\Documents\TensorFlow``).
- From your `Anaconda/Command Prompt` ``cd`` into the ``TensorFlow`` directory.
- To download the models you can either use `Git <https://git-scm.com/downloads>`_ to clone the `TensorFlow Models repo <https://github.com/tensorflow/models>`_ inside the ``TensorFlow`` folder, or you can simply download it as a `ZIP <https://github.com/tensorflow/models/archive/master.zip>`_ and extract it's contents inside the ``TensorFlow`` folder. To keep things consistent, in the latter case you will have to rename the extracted folder ``models-master`` to ``models``. [#]_
- You should now have a single folder named ``models`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

| TensorFlow
| └─ models
|     ├── official
|     ├── research
|     ├── samples
|     └── tutorials
|
|

.. [#] The latest repo commit when writing this tutorial is `da903e0 <https://github.com/tensorflow/models/commit/da903e07aea0887d59ebf612557243351ddfb4e6>`_.

Adding necessary Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since a lot of the scripts we will use require packages from ``Tensorflow\models\research\object_detection`` to be run, I have found that it's convenient to add the specific folder to our environmental variables.

For Linux users, this can be done by either adding to ``~/.bashrc`` or running the following code:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection

For Windows users, the following folder must be added to your ``Path`` environment variable (See :ref:`set_env`):

- ``<PATH_TO_TF>\TensorFlow\models\research\object_detection``

For whatever reason, some of the TensorFlow packages that we will need to use to do object detection, do not come pre-installed with our tensorflow installation. 

For Linux users ONLY, the `Installation docs <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>`_ suggest that you either run, or add to ``~/.bashrc`` file, the following command, which adds these packages to your PYTHONPATH:

.. code-block:: bash

    # From tensorflow/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

For Windows, the only way that I found works best, is to simply add the following folders to your ``Path`` environment variable (See also :ref:`set_env`):

- ``<PATH_TO_TF>\TensorFlow\models\research\slim``
- ``<PATH_TO_TF>\TensorFlow\models\research\slim\datasets``
- ``<PATH_TO_TF>\TensorFlow\models\research\slim\deployment``
- ``<PATH_TO_TF>\TensorFlow\models\research\slim\nets``
- ``<PATH_TO_TF>\TensorFlow\models\research\slim\preprocessing``
- ``<PATH_TO_TF>\TensorFlow\models\research\slim\scripts``

where ``<PATH_TO_TF>`` replaces the absolute path to your ``TesnorFlow`` folder. (e.g. ``<PATH_TO_TF>`` = ``C:\Users\sglvladi\Documents`` if ``TensorFlow`` resides within your ``Documents`` folder)

Protobuf Installation/Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be downloaded and compiled. 

This should be done as follows:

- Head to the `protoc releases page <https://github.com/google/protobuf/releases>`_
- Download the latest ``*-win32.zip`` release (e.g. ``protoc-3.5.1-win32.zip``)
- Create a folder in ``C:\Program Files`` and name it ``Google Protobuf``.
- Extract the contents of the downloaded ``*-win32.zip``, inside ``C:\Program Files\Google Protobuf``
- Add ``C:\Program Files\Google Protobuf\bin`` to your ``Path`` environment variable (see :ref:`set_env`)
- In a new `Anaconda/Command Prompt` [#]_, ``cd`` into ``TensorFlow/models/research/`` directory and run the following command:

    .. code-block:: python

        # From TensorFlow/models/research/
        protoc object_detection/protos/*.proto --python_out=.

If you are on Windows and using version 3.5 or later, the wildcard will not work and you have to run this in the command prompt:

.. code-block:: python

        # From TensorFlow/models/research/
        for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.

.. [#] NOTE: You MUST open a new `Anaconda/Command Prompt` for the changes in the environment variables to take effect.

.. _test_tf_models:

Test your Installation
~~~~~~~~~~~~~~~~~~~~~~

- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_gpu` environment (if you have not done so already) 
- ``cd`` into ``TensorFlow\models\research\object_detection`` and run the following command:

    .. code-block:: posh

        # From TensorFlow/models/research/object_detection
        jupyter notebook

- This should start a new ``jupyter notebook`` server on your machine and you should be redirected to a new tab of your default browser.
- Once there, simply follow `sentdex's Youtube video <https://youtu.be/COlbP62-B-U?t=7m23s>`_ to ensure that everything is running smoothly.
- If, when you try to run ``In [11]:``, Python crashes, have a look at the `Anaconda/Command Prompt` window you used to run the ``jupyter notebook`` service and check for a line similar (maybe identical) to the one below:

    .. code-block:: python

        2018-03-22 03:07:54.623130: E C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\cuda\cuda_dnn.cc:378] Loaded runtime CuDNN library: 7101 (compatibility version 7100) but source was compiled with 7003 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.

- If the above line is present in the printed debugging, it means that you have not installed the correct version of the cuDNN libraries. In this case make sure you re-do the :ref:`cudnn_install` step, making sure you instal cuDNN v7.0.5.

.. _labelImg_install:

LabelImg Installation
---------------------

For Windows and Linux you can download the precompiled binary at http://tzutalin.github.io/labelImg/.
The steps for installing from source follow below.

Create a new Conda virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To deal with the fact that ``labelImg`` (on Windows) requires the use of ``pyqt4``, while ``tensorflow 1.6`` (and possibly other packages) require ``pyqt5``, we will create a new virtual environment in which to run ``labelImg``.

* Open a new `Anaconda/Command Prompt` window 
* Type the following command:

    .. code-block:: posh

        conda create -n labelImg pyqt=4

* The above will create a new virtual environment with name ``labelImg``
* Now lets activate the newly created virtual environment by running the following in the `Anaconda Promt` window:

    .. code-block:: posh

        activate labelImg

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beginning of your cmd path specifier, e.g.:

.. code-block:: ps1con

    (labelImg) C:\Users\sglvladi> 

Downloading labelImg
~~~~~~~~~~~~~~~~~~~~

- Inside you ``TensorFlow`` folder, create a new directory, name it ``addons`` and then ``cd`` into it.
- To download the package you can either use `Git <https://git-scm.com/downloads>`_ to clone the `labelImg repo <https://github.com/tzutalin/labelImg>`_ inside the ``TensorFlow\addons`` folder, or you can simply download it as a `ZIP <https://github.com/tzutalin/labelImg/archive/master.zip>`_ and extract it's contents inside the ``TensorFlow\addons`` folder. To keep things consistent, in the latter case you will have to rename the extracted folder ``labelImg-master`` to ``labelImg``. [#]_
- You should now have a single folder named ``addons\labelImg`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

| TensorFlow
| ├─ addons
| │   └── labelImg
| └─ models
|     ├── official
|     ├── research
|     ├── samples
|     └── tutorials
|
|

.. [#] The latest repo commit when writing this tutorial is `8d1bd68 <https://github.com/tzutalin/labelImg/commit/8d1bd68ab66e8c311f2f45154729bba301a81f0b>`_.

Installing dependencies and compiling package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_gpu` environment (if you have not done so already) 
- ``cd`` into ``TensorFlow\addons\labelImg`` and run the following commands:

    .. code-block:: posh
        
        conda install pyqt=4
        conda install lxml
        pyrcc4 -py3 -o resources.py resources.qrc


Test your installation
~~~~~~~~~~~~~~~~~~~~~~

- Open a new `Anaconda/Command Prompt` window and activate the `tensorflow_gpu` environment (if you have not done so already) 
- ``cd`` into ``TensorFlow\addons\labelImg`` and run the following command:

    .. code-block:: posh
        
        python labelImg.py
        # or       
        python  labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]



