.. TensorFlow setup documentation master file, created by
   sphinx-quickstart on Wed Mar 21 19:03:08 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorFlow Object Detection API tutorial
============================================

.. important:: This tutorial is intended for TensorFlow 1.14, which (at the time of writing this tutorial) is the latest stable version before TensorFlow 2.x.

   Tensorflow 1.15 has also been released, but seems to be exhibiting `instability issues <https://github.com/tensorflow/models/issues/7640>`_.

   A version for TensorFlow 1.9 can be found `here <https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/v1.9.1/>`_.

   At the time of righting this tutorial, Object Detection model training and evaluation was not migrated to TensorFlow 2.x (see `here <https://github.com/tensorflow/models/issues/6423#issuecomment-600925072>`_). From personal tests, it seems that detection using pre-trained models works, however it is not yet possible to train and evaluate models. Once the migration has been completed, a version for TensorFlow 2.x will be produced.

This is a step-by-step tutorial/guide to setting up and using TensorFlow's Object Detection API to perform, namely, object detection in images/video.

The software tools which we shall use throughout this tutorial are listed in the table below:

+---------------------------------------------+
| Target Software versions                    |
+==============+==============================+
| OS           | Windows, Linux               |
+--------------+------------------------------+
| Python       | 3.7                          |
+--------------+------------------------------+
| TensorFlow   | 1.14                         |
+--------------+------------------------------+
| CUDA Toolkit | 10.0                         |
+--------------+------------------------------+
| CuDNN        | 7.6.5                        |
+--------------+------------------------------+ 
| Anaconda     | Python 3.7 (Optional)        |
+--------------+------------------------------+

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   install
   camera
   training
   issues



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
