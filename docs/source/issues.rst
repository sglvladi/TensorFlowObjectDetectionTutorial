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

labelImg saves annotation files with ``.xml.xml`` extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the time of writing up this document, I haven't managed to identify why this might be happening. I have joined a `GitHub issue <https://github.com/tzutalin/labelImg/issues/252>`_, at which you can refer in case there are any updates.

One way I managed to fix the issue was by clicking on the "Change Save Dir" button and selecting the directory where the annotations files should be stores. By doing so, you should not longer get a pop-up dialog when you click "Save" (or Ctrl+s), but you can always check if the file was saved by looking at the bottom left corner of ``labelImg``.