.. TensorFlow setup documentation master file, created by
   sphinx-quickstart on Wed Mar 21 19:03:08 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorFlow 2 Object Detection API tutorial
==========================================

.. important:: This tutorial is intended for TensorFlow 2.5, which (at the time of writing this tutorial) is the latest stable version of TensorFlow 2.x.

   A version for TensorFlow 2.2 can be found `here <https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/2.2.0/>`_.

   A version for TensorFlow 1.14 can be found `here <https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/>`_.

This is a step-by-step tutorial/guide to setting up and using TensorFlow's Object Detection API to perform, namely, object detection in images/video.

The software tools which we shall use throughout this tutorial are listed in the table below:

+---------------------------------------------+
| Target Software versions                    |
+==============+==============================+
| OS           | Windows, Linux               |
+--------------+------------------------------+
| Python       | 3.9 [#]_                     |
+--------------+------------------------------+
| TensorFlow   | 2.5.0                        |
+--------------+------------------------------+
| CUDA Toolkit | 11.2                         |
+--------------+------------------------------+
| CuDNN        | 8.1.0                        |
+--------------+------------------------------+ 
| Anaconda     | Python 3.8 (Optional)        |
+--------------+------------------------------+

.. [#] Python 3.9 is not a strict requirement. Any Python 3.x version should work, although this has not been tested.


.. toctree::
   :maxdepth: 4
   :caption: Contents

   install
   training
   auto_examples/index
   issues
