.. _issues:

Common issues
=============

Below is a list of common issues encountered while using TensorFlow for objects detection.

Python crashes - TensorFlow GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using :ref:`tensorflow_gpu` and when you try to run some Python object detection script (e.g. :ref:`test_tf_models`), after a few seconds, Windows reports that Python has crashed then have a look at the `Anaconda/Command Prompt` window you used to run the script and check for a line similar (maybe identical) to the one below:

    .. code-block:: python

        2018-03-22 03:07:54.623130: E C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\cuda\cuda_dnn.cc:378] Loaded runtime CuDNN library: 7101 (compatibility version 7100) but source was compiled with 7003 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.

If the above line is present in the printed debugging, it means that you have not installed the correct version of the cuDNN libraries. In this case make sure you re-do the :ref:`cudnn_install` step, making sure you instal cuDNN v7.0.5. 

Cleaning up Nvidia containers (TensorFlow GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, when terminating a TensorFlow training process, the Nvidia containers associated to the process are not cleanly terminated. This can lead to bogus errors when we try to run a new TensorFlow process.

Some known issues caused by the above are presented below:

- Failure to restart training of a model. Look for the following errors in the debugging:

    .. code-block:: python

        2018-03-23 03:03:10.326902: E C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\cuda\cuda_dnn.cc:385] could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED
        2018-03-23 03:03:10.330475: E C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\cuda\cuda_dnn.cc:352] could not destroy cudnn handle: CUDNN_STATUS_BAD_PARAM
        2018-03-23 03:03:10.333797: W C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow/stream_executor/stream.h:1983] attempting to perform DNN operation using StreamExecutor without DNN support
        2018-03-23 03:03:10.333807: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\stream.cc:1851] stream 00000216F05CB660 did not wait for stream: 00000216F05CA6E0
        2018-03-23 03:03:10.340765: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\stream.cc:4637] stream 00000216F05CB660 did not memcpy host-to-device; source: 000000020DB37B00
        2018-03-23 03:03:10.343752: F C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_util.cc:343] CPU->GPU Memcpy failed

To solve such issues in Windows, open a `Task Manager` windows, look for Tasks with name ``NVIDIA Container`` and kill them by selecting them and clicking the `End Task` button at the bottom left corner of the window.

If the issue persists, then you're probably running out of memory. Try closing down anything else that might be eating up your GPU memory (e.g. Youtube videos, webpages etc.)

"WARNING:tensorflow:Entity ``<bound method X of <Y>>`` could not be transformed ..."
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some versions of Tensorflow, you may see errors that look similar to the ones below:

.. code-block:: python

    ...
    WARNING:tensorflow:Entity <bound method Conv.call of <tensorflow.python.layers.convolutional.Conv2D object at 0x000001E92103EDD8>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Conv.call of <tensorflow.python.layers.convolutional.Conv2D object at 0x000001E92103EDD8>>: AssertionError: Bad argument number for Name: 3, expecting 4
    WARNING:tensorflow:Entity <bound method BatchNormalization.call of <tensorflow.python.layers.normalization.BatchNormalization object at 0x000001E9225EBA90>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method BatchNormalization.call of <tensorflow.python.layers.normalization.BatchNormalization object at 0x000001E9225EBA90>>: AssertionError: Bad argument number for Name: 3, expecting 4
    ...

These warnings appear to be harmless form my experience, however they can saturate the console with unnecessary messages, which makes it hard to scroll through the output of the training/evaluation process.

As reported `here <https://github.com/tensorflow/tensorflow/issues/34551>`_, this issue seems to
be caused by a mismatched version of `gast <https://github.com/serge-sans-paille/gast/>`_. Simply
downgrading gast to version ``0.2.2`` seems to remove the warnings. This can be done by running:

.. code-block:: bash

    pip install gast==0.2.2

"AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is possible that when executing ``from object_detection.utils import label_map_util`` you may
get the above error. As per the discussion is in `this Stack Overflow thread <https://stackoverflow.com/a/61961016/3474873>`_,
upgrading the Python protobuf version seems to solve this issue:

.. code-block:: bash

    pip install --upgrade protobuf

.. _export_error:

"TypeError: Expected Operation, Variable, or Tensor, got level_5"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When trying to export oyu trained model using the ``exporter_main_v2.py`` script, you may come
across an error that looks like this:

.. code-block:: bash
    :linenos:
    :emphasize-lines: 9

    Traceback (most recent call last):
      File ".\exporter_main_v2.py", line 126, in <module>
        app.run(main)
      File "C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\absl\app.py", line 299, in run
        _run_main(main, args)
      ...
      File "C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\tensorflow\python\keras\engine\base_layer.py", line 1627, in get_losses_for
        reachable = tf_utils.get_reachable_from_inputs(inputs, losses)
      File "C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\tensorflow\python\keras\utils\tf_utils.py", line 140, in get_reachable_from_inputs
        raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))
    TypeError: Expected Operation, Variable, or Tensor, got level_5

This error seems to come from TensorFlow itself and a discussion on the issue can be found
`here <https://github.com/tensorflow/models/issues/8841>`_. As discussed there, a fix to the above
issue can be achieved by opening the ``tf_utils.py`` file and adding a line of code. Below is a
summary of how this can be done:

- Look at the line that corresponds to line 9 (highlighted) in the above error print out.
- Copy the path to the ``tf_utils.py`` file; in my case this was ``C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\tensorflow\python\keras\utils\tf_utils.py``
- Open the file and replace line 140 of the file as follows:

  - Change:

    .. code-block:: python

        raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))

    to:

    .. code-block:: python

        if not isinstance(x, str):
            raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))

At the time of writting this tutorial, a fix to the issue had not been implemented in the version
of TensorFlow installed using ``pip``. It is possible that this will get incorporated at some later
point.