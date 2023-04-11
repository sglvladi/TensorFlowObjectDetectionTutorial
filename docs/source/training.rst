Training Custom Object Detector
===============================

So, up to now you should have done the following:

- Installed TensorFlow (See :ref:`tf_install`)
- Installed TensorFlow Object Detection API (See :ref:`tf_models_install`)

Now that we have done all the above, we can start doing some cool stuff. Here we will see how you can train your own object detector, and since it is not as simple as it sounds, we will have a look at:

1. How to organise your workspace/training files
2. How to prepare/annotate image datasets
3. How to generate tf records from such datasets
4. How to configure a simple training pipeline 
5. How to train a model and monitor it's progress
6. How to export the resulting model and use it to detect objects.

Preparing the Workspace
-----------------------

1. If you have followed the tutorial, you should by now have a folder ``Tensorflow``, placed under ``<PATH_TO_TF>`` (e.g. ``C:/Users/sglvladi/Documents``), with the following directory tree:

    .. code-block:: default

        TensorFlow/
        ├─ addons/ (Optional)
        │  └─ labelImg/
        └─ models/
           ├─ community/
           ├─ official/
           ├─ orbit/
           ├─ research/
           └─ ...

2. Now create a new folder under ``TensorFlow``  and call it ``workspace``. It is within the ``workspace`` that we will store all our training set-ups. Now let's go under workspace and create another folder named ``training_demo``. Now our directory structure should be as so:

    .. code-block:: default

        TensorFlow/
        ├─ addons/ (Optional)
        │  └─ labelImg/
        ├─ models/
        │  ├─ community/
        │  ├─ official/
        │  ├─ orbit/
        │  ├─ research/
        │  └─ ...
        └─ workspace/
           └─ training_demo/

3. The ``training_demo`` folder shall be our `training folder`, which will contain all files related to our model training. It is advisable to create a separate training folder each time we wish to train on a different dataset. The typical structure for training folders is shown below.

    .. code-block:: default
    
        training_demo/
        ├─ annotations/
        ├─ exported-models/
        ├─ images/
        │  ├─ test/
        │  └─ train/
        ├─ models/
        ├─ pre-trained-models/
        └─ README.md

Here's an explanation for each of the folders/filer shown in the above tree:

- ``annotations``: This folder will be used to store all ``*.csv`` files and the respective TensorFlow ``*.record`` files, which contain the list of annotations for our dataset images. 
- ``exported-models``: This folder will be used to store exported versions of our trained model(s).
- ``images``: This folder contains a copy of all the images in our dataset, as well as the respective ``*.xml`` files produced for each one, once ``labelImg`` is used to annotate objects.

    * ``images/train``: This folder contains a copy of all images, and the respective ``*.xml`` files, which will be used to train our model.
    * ``images/test``: This folder contains a copy of all images, and the respective ``*.xml`` files, which will be used to test our model.

- ``models``: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file ``*.config``, as well as all files generated during the training and evaluation of our model.
- ``pre-trained-models``: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.
- ``README.md``: This is an optional file which provides some general information regarding the training conditions of our model. It is not used by TensorFlow in any way, but it generally helps when you have a few training folders and/or you are revisiting a trained model after some time.

If you do not understand most of the things mentioned above, no need to worry, as we'll see how all the files are generated further down.


Preparing the Dataset
---------------------

Annotate the Dataset
********************

.. _labelImg_install:

Install LabelImg
~~~~~~~~~~~~~~~~

There exist several ways to install ``labelImg``. Below are 3 of the most common.

.. _labelImg_install_pip:

Using PIP (Recommended)
#######################
1. Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
2. Run the following command to install ``labelImg``:

.. code-block:: default

    pip install labelImg

3. ``labelImg`` can then be run as follows:

.. code-block:: default

    labelImg
    # or
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

Use precompiled binaries (Easy)
###############################
Precompiled binaries for both Windows and Linux can be found `here <http://tzutalin.github.io/labelImg/>`__ .

Installation is the done in three simple steps:

1. Inside you ``TensorFlow`` folder, create a new directory, name it ``addons`` and then ``cd`` into it.

2. Download the latest binary for your OS from `here <http://tzutalin.github.io/labelImg/>`__. and extract its contents under ``Tensorflow/addons/labelImg``.

3. You should now have a single folder named ``addons/labelImg`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

.. code-block:: default

    TensorFlow/
    ├─ addons/
    │  └─ labelImg/
    └─ models/
       ├─ community/
       ├─ official/
       ├─ orbit/
       ├─ research/
       └─ ...

4. ``labelImg`` can then be run as follows:

.. code-block:: default

    # From within Tensorflow/addons/labelImg
    labelImg
    # or
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

Build from source (Hard)
########################
The steps for installing from source follow below.

**1. Download labelImg**

- Inside you ``TensorFlow`` folder, create a new directory, name it ``addons`` and then ``cd`` into it.
- To download the package you can either use `Git <https://git-scm.com/downloads>`_ to clone the `labelImg repo <https://github.com/tzutalin/labelImg>`_ inside the ``TensorFlow\addons`` folder, or you can simply download it as a `ZIP <https://github.com/tzutalin/labelImg/archive/master.zip>`_ and extract it's contents inside the ``TensorFlow\addons`` folder. To keep things consistent, in the latter case you will have to rename the extracted folder ``labelImg-master`` to ``labelImg``. [#]_
- You should now have a single folder named ``addons\labelImg`` under your ``TensorFlow`` folder, which contains another 4 folders as such:

.. code-block:: default

    TensorFlow/
    ├─ addons
    │  └─ labelImg/
    └─ models/
       ├─ community/
       ├─ official/
       ├─ orbit/
       ├─ research/
       └─ ...

.. [#] The latest repo commit when writing this tutorial is `8d1bd68 <https://github.com/tzutalin/labelImg/commit/8d1bd68ab66e8c311f2f45154729bba301a81f0b>`_.

**2. Install dependencies and compiling package**

- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- ``cd`` into ``TensorFlow/addons/labelImg`` and run the following commands:

    .. tabs::

        .. tab:: Windows

            .. code-block:: default

                conda install pyqt=5
                pyrcc5 -o libs/resources.py resources.qrc

        .. tab:: Linux

            .. code-block:: default

                sudo apt-get install pyqt5-dev-tools
                sudo pip install -r requirements/requirements-linux-python3.txt
                make qt5py3


**3. Test your installation**

- Open a new `Terminal` window and activate the `tensorflow_gpu` environment (if you have not done so already)
- ``cd`` into ``TensorFlow/addons/labelImg`` and run the following command:

    .. code-block:: default

        # From within Tensorflow/addons/labelImg
        python labelImg.py
        # or
        python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]


Annotate Images
~~~~~~~~~~~~~~~

- Once you have collected all the images to be used to test your model (ideally more than 100 per class), place them inside the folder ``training_demo/images``.
- Open a new `Terminal` window.
- Next go ahead and start ``labelImg``, pointing it to your ``training_demo/images`` folder.

   - If you installed ``labelImg`` :ref:`labelImg_install_pip`:

    .. code-block:: default

        labelImg <PATH_TO_TF>/TensorFlow/workspace/training_demo/images

   - Othewise, ``cd`` into ``Tensorflow/addons/labelImg`` and run:

    .. code-block:: default

        # From within Tensorflow/addons/labelImg
        python labelImg.py ../../workspace/training_demo/images

- A File Explorer Dialog windows should open, which points to the ``training_demo/images`` folder.
- Press the "Select Folder" button, to start annotating your images.

Once open, you should see a window similar to the one below:

.. image:: ./_static/labelImg.JPG
   :width: 90%
   :alt: alternate text
   :align: center

I won't be covering a tutorial on how to use ``labelImg``, but you can have a look at `labelImg's repo <https://github.com/tzutalin/labelImg#usage>`_ for more details. A nice Youtube video demonstrating how to use ``labelImg`` is also available `here <https://youtu.be/K_mFnvzyLvc?t=9m13s>`__. What is important is that once you annotate all your images, a set of new ``*.xml`` files, one for each image, should be generated inside your ``training_demo/images`` folder.


.. _image_partitioning_sec:

Partition the Dataset
*********************

Once you have finished annotating your image dataset, it is a general convention to use only part
of it for training, and the rest is used for evaluation purposes (e.g. as discussed in
:ref:`evaluation_sec`).

Typically, the ratio is 9:1, i.e. 90% of the images are used for training and the rest 10% is
maintained for testing, but you can chose whatever ratio suits your needs.

Once you have decided how you will be splitting your dataset, copy all training images, together
with their corresponding ``*.xml`` files, and place them inside the ``training_demo/images/train``
folder. Similarly, copy all testing images, with their ``*.xml`` files, and paste them inside
``training_demo/images/test``.

For lazy people like myself, who cannot be bothered to do the above, I have put together a simple
script that automates the above process:

.. literalinclude:: scripts/partition_dataset.py

- Under the ``TensorFlow`` folder, create a new folder ``TensorFlow/scripts``, which we can use to store some useful scripts.
- To make things even tidier, let's create a new folder ``TensorFlow/scripts/preprocessing``, where we shall store scripts that we can use to preprocess our training inputs. Below is out ``TensorFlow`` directory tree structure, up to now:

    .. code-block:: default

        TensorFlow/
        ├─ addons/ (Optional)
        │  └─ labelImg/
        ├─ models/
        │  ├─ community/
        │  ├─ official/
        │  ├─ orbit/
        │  ├─ research/
        │  └─ ...
        ├─ scripts/
        │  └─ preprocessing/
        └─ workspace/
           └─ training_demo/

- Click :download:`here <scripts/partition_dataset.py>` to download the above script and save it inside ``TensorFlow/scripts/preprocessing``.
- Then,  ``cd`` into ``TensorFlow/scripts/preprocessing`` and run:

    .. code-block:: default

        python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1

        # For example
        # python partition_dataset.py -x -i C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images -r 0.1

Once the script has finished, two new folders should have been created under ``training_demo/images``,
namely ``training_demo/images/train`` and ``training_demo/images/test``, containing 90% and 10% of
the images (and ``*.xml`` files), respectively. To avoid loss of any files, the script will not
delete the images under ``training_demo/images``. Once you have checked that your images have been
safely copied over, you can delete the images under ``training_demo/images`` manually.


Create Label Map
****************

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

Below we show an example label map (e.g ``label_map.pbtxt``), assuming that our dataset contains 2 labels, ``dogs`` and ``cats``:

.. code-block:: default

    item {
        id: 1
        name: 'cat'
    }

    item {
        id: 2
        name: 'dog'
    }

Label map files have the extention ``.pbtxt`` and should be placed inside the ``training_demo/annotations`` folder.

Create TensorFlow Records
*************************

Now that we have generated our annotations and split our dataset into the desired training and
testing subsets, it is time to convert our annotations into the so called ``TFRecord`` format.


Convert ``*.xml`` to ``*.record``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To do this we can write a simple script that iterates through all ``*.xml`` files in the
``training_demo/images/train`` and ``training_demo/images/test`` folders, and generates a
``*.record`` file for each of the two. Here is an example script that allows us to do just that:

.. literalinclude:: scripts/generate_tfrecord.py


- Click :download:`here <scripts/generate_tfrecord.py>` to download the above script and save it inside ``TensorFlow/scripts/preprocessing``.
- Install the ``pandas`` package:

    .. code-block:: default

        conda install pandas # Anaconda
                             # or
        pip install pandas   # pip

- Finally, ``cd`` into ``TensorFlow/scripts/preprocessing`` and run:

    .. code-block:: default
        
        # Create train data:
        python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

        # Create test data:
        python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

        # For example
        # python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/train -l C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/train.record
        # python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/test -l C:/Users/sglvladi/Documents/Tensorflow2/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/test.record

Once the above is done, there should be 2 new files under the ``training_demo/annotations`` folder, named ``test.record`` and ``train.record``, respectively.


.. _config_training_pipeline_sec:

Configuring a Training Job
--------------------------

For the purposes of this tutorial we will not be creating a training job from scratch, but rather
we will reuse one of the pre-trained models provided by TensorFlow. If you would like to train an
entirely new model, you can have a look at `TensorFlow's tutorial <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md>`_.

The model we shall be using in our examples is the `SSD ResNet50 V1 FPN 640x640 <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>`_
model, since it provides a relatively good trade-off between performance and speed. However, there
exist a number of other models you can use, all of which are listed in `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.

Download Pre-Trained Model
**************************
To begin with, we need to download the latest pre-trained network for the model we wish to use.
This can be done by simply clicking on the name of the desired model in the table found in
`TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
Clicking on the name of your model should initiate a download for a ``*.tar.gz`` file.

Once the ``*.tar.gz`` file has been downloaded, open it using a decompression program of your
choice (e.g. 7zip, WinZIP, etc.). Next, open the ``*.tar`` folder that you see when the compressed
folder is opened, and extract its contents inside the folder ``training_demo/pre-trained-models``.
Since we downloaded the `SSD ResNet50 V1 FPN 640x640 <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>`_
model, our ``training_demo`` directory should now look as follows:

 .. code-block:: default

        training_demo/
        ├─ ...
        ├─ pre-trained-models/
        │  └─ ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
        │     ├─ checkpoint/
        │     ├─ saved_model/
        │     └─ pipeline.config
        └─ ...

Note that the above process can be repeated for all other pre-trained models you wish to experiment
with. For example, if you wanted to also configure a training job for the `EfficientDet D1 640x640 <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz>`_
model, you can download the model and after extracting its context the demo directory will be:

 .. code-block:: default

        training_demo/
        ├─ ...
        ├─ pre-trained-models/
        │  ├─ efficientdet_d1_coco17_tpu-32/
        │  │  ├─ checkpoint/
        │  │  ├─ saved_model/
        │  │  └─ pipeline.config
        │  └─ ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
        │     ├─ checkpoint/
        │     ├─ saved_model/
        │     └─ pipeline.config
        └─ ...

.. _training_pipeline_conf:

Configure the Training Pipeline
*******************************
Now that we have downloaded and extracted our pre-trained model, let's create a directory for our
training job. Under the ``training_demo/models`` create a new directory named ``my_ssd_resnet50_v1_fpn``
and copy the ``training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config``
file inside the newly created directory. Our ``training_demo/models`` directory should now look
like this:

 .. code-block:: default

        training_demo/
        ├─ ...
        ├─ models/
        │  └─ my_ssd_resnet50_v1_fpn/
        │     └─ pipeline.config
        └─ ...

Now, let's have a look at the changes that we shall need to apply to the ``pipeline.config`` file
(highlighted in yellow):

.. literalinclude:: scripts/pipeline.config
    :linenos:
    :emphasize-lines: 3,131,161,167,168,172,174,178,179,182,186

It is worth noting here that the changes to lines ``178`` to ``179`` above are optional. These
should only be used if you installed the COCO evaluation tools, as outlined in the
:ref:`tf_models_install_coco` section, and you intend to run evaluation (see :ref:`evaluation_sec`).

Once the above changes have been applied to our config file, go ahead and save it.

.. _training_sec:

Training the Model
------------------
Before we begin training our model, let's go and copy the ``TensorFlow/models/research/object_detection/model_main_tf2.py``
script and paste it straight into our ``training_demo`` folder. We will need this script in order
to train our model.

Now, to initiate a new training job, open a new `Terminal`,  ``cd`` inside the ``training_demo``
folder and run the following command:

.. code-block:: default

    python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

Once the training process has been initiated, you should see a series of print outs similar to the
one below (plus/minus some warnings):

.. code-block:: default

    ...
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
    W0716 05:24:19.105542  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
    W0716 05:24:19.106541  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
    W0716 05:24:19.107540  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
    W0716 05:24:19.108539  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    W0716 05:24:19.108539  1364 util.py:151] A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    INFO:tensorflow:Step 100 per-step time 1.153s loss=0.761
    I0716 05:26:55.879558  1364 model_lib_v2.py:632] Step 100 per-step time 1.153s loss=0.761
    ...

.. important::

    The output will normally look like it has "frozen", but DO NOT rush to cancel the process. The
    training outputs logs only every 100 steps by default, therefore if you wait for a while, you
    should see a log for the loss at step 100.

    The time you should wait can vary greatly, depending on whether you are using a GPU and the
    chosen value for ``batch_size`` in the config file, so be patient.

If you ARE observing a similar output to the above, then CONGRATULATIONS, you have successfully
started your first training job. Now you may very well treat yourself to a cold beer, as waiting
on the training to finish is likely to take a while. Following what people have said online, it
seems that it is advisable to allow you model to reach a ``TotalLoss`` of at least 2 (ideally 1
and lower) if you want to achieve "fair" detection results. Obviously, lower ``TotalLoss`` is
better, however very low ``TotalLoss`` should be avoided, as the model may end up overfitting the
dataset, meaning that it will perform poorly when applied to images outside the dataset. To
monitor ``TotalLoss``, as well as a number of other metrics, while your model is training, have a
look at :ref:`tensorboard_sec`.

If you ARE NOT seeing a print-out similar to that shown above, and/or the training job crashes
after a few seconds, then have a look at the issues and proposed solutions, under the
:ref:`issues` section, to see if you can find a solution. Alternatively, you can try the issues
section of the official `Tensorflow Models repo <https://github.com/tensorflow/models/issues>`_.

.. note::
    Training times can be affected by a number of factors such as:

    - The computational power of you hardware (either CPU or GPU): Obviously, the more powerful your PC is, the faster the training process.
    - Whether you are using the TensorFlow CPU or GPU variant: In general, even when compared to the best CPUs, almost any GPU graphics card will yield much faster training and detection speeds. As a matter of fact, when I first started I was running TensorFlow on my `Intel i7-5930k` (6/12 cores @ 4GHz, 32GB RAM) and was getting step times of around `12 sec/step`, after which I installed TensorFlow GPU and training the very same model -using the same dataset and config files- on a `EVGA GTX-770` (1536 CUDA-cores @ 1GHz, 2GB VRAM) I was down to `0.9 sec/step`!!! A 12-fold increase in speed, using a "low/mid-end" graphics card, when compared to a "mid/high-end" CPU.
    - The complexity of the objects you are trying to detect: Obviously, if your objective is to track a black ball over a white background, the model will converge to satisfactory levels of detection pretty quickly. If on the other hand, for example, you wish to detect ships in ports, using Pan-Tilt-Zoom cameras, then training will be a much more challenging and time-consuming process, due to the high variability of the shape and size of ships, combined with a highly dynamic background.
    - And many, many, many, more....

.. _evaluation_sec:

Evaluating the Model (Optional)
-------------------------------

By default, the training process logs some basic measures of training performance. These seem to
change depending on the installed version of Tensorflow.

As you will have seen in various parts of this tutorial, we have mentioned a few times the
optional utilisation of the COCO evaluation metrics. Also, under section
:ref:`image_partitioning_sec` we partitioned our dataset in two parts, where one was to be used
for training and the other for evaluation. In this section we will look at how we can use these
metrics, along with the test images, to get a sense of the performance achieved by our model as it
is being trained.

Firstly, let's start with a brief explanation of what the evaluation process does. While the
training process runs, it will occasionally create checkpoint files inside the
``training_demo/training`` folder, which correspond to snapshots of the model at given steps. When
a set of such new checkpoint files is generated, the evaluation process uses these files and
evaluates how well the model performs in detecting objects in the test dataset. The results of
this evaluation are summarised in the form of some metrics, which can be examined over time.

The steps to run the evaluation are outlined below:

#. Firstly we need to download and install the metrics we want to use.

   - For a description of the supported object detection evaluation metrics, see `here <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md>`__.
   - The process of installing the COCO evaluation metrics is described in :ref:`tf_models_install_coco`.

#. Secondly, we must modify the configuration pipeline (``*.config`` script).

   - See lines 178-179 of the script in :ref:`training_pipeline_conf`.

#. The third step is to actually run the evaluation. To do so, open a new `Terminal`,  ``cd`` inside the ``training_demo`` folder and run the following command:

    .. code-block:: default

        python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn

    Once the above is run, you should see a checkpoint similar to the one below  (plus/minus some warnings):

    .. code-block:: default

        ...
        WARNING:tensorflow:From C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\object_detection\inputs.py:79: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
        Instructions for updating:
        Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
        W0716 05:44:10.059399 17144 deprecation.py:317] From C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\object_detection\inputs.py:79: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
        Instructions for updating:
        Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
        WARNING:tensorflow:From C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\object_detection\inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
        Instructions for updating:
        Use `tf.cast` instead.
        W0716 05:44:12.383937 17144 deprecation.py:317] From C:\Users\sglvladi\Anaconda3\envs\tf2\lib\site-packages\object_detection\inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
        Instructions for updating:
        Use `tf.cast` instead.
        INFO:tensorflow:Waiting for new checkpoint at models/my_ssd_resnet50_v1_fpn
        I0716 05:44:22.779590 17144 checkpoint_utils.py:125] Waiting for new checkpoint at models/my_ssd_resnet50_v1_fpn
        INFO:tensorflow:Found new checkpoint at models/my_ssd_resnet50_v1_fpn\ckpt-2
        I0716 05:44:22.882485 17144 checkpoint_utils.py:134] Found new checkpoint at models/my_ssd_resnet50_v1_fpn\ckpt-2

While the evaluation process is running, it will periodically check (every 300 sec by default) and
use the latest ``models/my_ssd_resnet50_v1_fpn/ckpt-*`` checkpoint files to evaluate the performance
of the model. The results are stored in the form of tf event files (``events.out.tfevents.*``)
inside ``models/my_ssd_resnet50_v1_fpn/eval_0``. These files can then be used to monitor the
computed metrics, using the process described by the next section.

.. _tensorboard_sec:

Monitor Training Job Progress using TensorBoard
-----------------------------------------------

A very nice feature of TensorFlow, is that it allows you to coninuously monitor and visualise a
number of different training/evaluation metrics, while your model is being trained. The specific
tool that allows us to do all that is `Tensorboard <https://www.tensorflow.org/tensorboard>`_.

To start a new TensorBoard server, we follow the following steps:

- Open a new `Anaconda/Command Prompt`
- Activate your TensorFlow conda environment (if you have one), e.g.:

    .. code-block:: default

        activate tensorflow_gpu

- ``cd`` into the ``training_demo`` folder.
- Run the following command:

    .. code-block:: default

        tensorboard --logdir=models/my_ssd_resnet50_v1_fpn

The above command will start a new TensorBoard server, which (by default) listens to port 6006 of
your machine. Assuming that everything went well, you should see a print-out similar to the one
below (plus/minus some warnings):

.. code-block:: default

    ...
    TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)

Once this is done, go to your browser and type ``http://localhost:6006/`` in your address bar,
following which you should be presented with a dashboard similar to the one shown below
(maybe less populated if your model has just started training):

.. image:: ./_static/TensorBoard.JPG
   :width: 90%
   :alt: alternate text
   :align: center



Exporting a Trained Model
-------------------------

Once your training job is complete, you need to extract the newly trained inference graph, which
will be later used to perform the object detection. This can be done as follows:

- Copy the ``TensorFlow/models/research/object_detection/exporter_main_v2.py`` script and paste it straight into your ``training_demo`` folder.
- Now, open a `Terminal`, ``cd`` inside your ``training_demo`` folder, and run the following command:

.. code-block:: default
    
    python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model

After the above process has completed, you should find a new folder ``my_model`` under the
``training_demo/exported-models``, that has the following structure:

    .. code-block:: default

        training_demo/
        ├─ ...
        ├─ exported-models/
        │  └─ my_model/
        │     ├─ checkpoint/
        │     ├─ saved_model/
        │     └─ pipeline.config
        └─ ...

This model can then be used to perform inference.

.. note::

    You may get the following error when trying to export your model:

    .. code-block:: default

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

    If this happens, have a look at the :ref:`export_error` issue section for a potential solution.

