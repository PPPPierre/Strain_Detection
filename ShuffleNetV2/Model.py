import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
tf.logging.set_verbosity('INFO')

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3

def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164
    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            x = block(x, num_units=8, scope='Stage3')
            x = block(x, num_units=4, scope='Stage4')

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits


def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x



SHUFFLE_BUFFER_SIZE = 10000
NUM_FILES_READ_IN_PARALLEL = 10
NUM_PARALLEL_CALLS = 8
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
IMAGE_SIZE = 224  # this will be used for training and evaluation
MIN_DIMENSION = 256  # when evaluating, resize to this size before doing central crop


class Pipeline:
    def __init__(self, filenames, is_training, batch_size, num_epochs):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size, num_epochs: integers.
        """
        self.is_training = is_training

        # read the files in parallel
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)
        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=NUM_FILES_READ_IN_PARALLEL
        ))
        dataset = dataset.prefetch(buffer_size=batch_size)

        # mix the training examples
        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(num_epochs)

        # decode and augment data
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            self.parse_and_preprocess, batch_size=batch_size,
            num_parallel_batches=1, drop_remainder=False
        ))
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.
        Returns:
            image: a float tensor with shape [height, width, 3],
                a RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image as a string, it will be decoded later
        image_as_string = parsed_features['image']

        # get a label
        label = tf.to_int32(parsed_features['label'])

        if self.is_training:

            # get groundtruth boxes, they must be in from-zero-to-one format,
            # also, it assumed that ymin < ymax and xmin < xmax
            boxes = tf.stack([
                parsed_features['ymin'], parsed_features['xmin'],
                parsed_features['ymax'], parsed_features['xmax']
            ], axis=1)
            boxes = tf.to_float(boxes)  # shape [num_boxes, 4]
            # they are only used for data augmentation

            image = self.augmentation(image_as_string, boxes)
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range
            image = resize_keeping_aspect_ratio(image, MIN_DIMENSION)
            image = central_crop(image, crop_height=IMAGE_SIZE, crop_width=IMAGE_SIZE)

        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

        # in the format required by tf.estimator,
        # they will be batched later
        features = {'images': image}
        labels = {'labels': label}
        return features, labels

    def augmentation(self, image_as_string, boxes):

        image = get_random_crop(image_as_string, boxes)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_images(
            image, [IMAGE_SIZE, IMAGE_SIZE],
            method=RESIZE_METHOD
        )

        image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range
        image = random_color_manipulations(image, probability=0.25, grayscale_probability=0.05)
        return image


def resize_keeping_aspect_ratio(image, min_dimension):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
    Returns:
        a float tensor with shape [new_height, new_width, 3],
            where min_dimension = min(new_height, new_width).
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension) / original_min_dim
    new_height = tf.round(height * scale_factor)
    new_width = tf.round(width * scale_factor)

    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    return image


def get_random_crop(image_as_string, boxes):

    distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_as_string),
        bounding_boxes=tf.expand_dims(boxes, axis=0),
        min_object_covered=0.25,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    begin, size, _ = distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    crop = tf.image.decode_and_crop_jpeg(
        image_as_string, crop_window, channels=3
    )
    return crop


def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def random_color_manipulations(image, probability=0.1, grayscale_probability=0.1, fast=True):

    def manipulate(image):
        if not fast:
            # intensity and order of this operations are kinda random,
            # so you will need to tune this for you problem
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.15)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0.0, 1.0)
        else:
            image = distort_color_fast(image)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)

    return image


def distort_color_fast(image):
    with tf.name_scope('distort_color'):
        br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
        cb_factor = tf.random_uniform([], -0.1, 0.1)
        cr_factor = tf.random_uniform([], -0.1, 0.1)
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = shufflenet(
        features['images'], is_training,
        num_classes=params['num_classes'],
        depth_multiplier=params['depth_multiplier']
    )
    predictions = {
        'probabilities': tf.nn.softmax(logits, axis=1),
        'classes': tf.argmax(logits, axis=1, output_type=tf.int32)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    with tf.name_scope('cross_entropy'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['labels'], logits=logits)
        loss = tf.reduce_mean(losses, axis=0)
        tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels['labels'], predictions['classes']),
            'top5_accuracy': tf.metrics.mean(tf.to_float(tf.nn.in_top_k(
                predictions=predictions['probabilities'], targets=labels['labels'], k=5
            )))
        }
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            params['decay_steps'], params['end_learning_rate'],
            power=1.0
        )  # linear decay
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.name_scope('evaluation_ops'):
        train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
            labels['labels'], predictions['classes']
        )), axis=0)
        train_top5_accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(
            predictions=predictions['probabilities'], targets=labels['labels'], k=5
        )), axis=0)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('train_top5_accuracy', train_top5_accuracy)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore
        )

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)


"""
The purpose of this script is to train a network.
Evaluation will happen periodically.
To use it just run:
python train.py
Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 512
NUM_EPOCHS = 133  # set 166 for 1.0x version
TRAIN_DATASET_SIZE = 1281144
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
PARAMS = {
    'train_dataset_path': '/mnt/datasets/imagenet/train_shards/',
    'val_dataset_path': '/mnt/datasets/imagenet/val_shards/',
    'weight_decay': 4e-5,
    'initial_learning_rate': 0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-6,
    'model_dir': 'models/run00/',
    'num_classes': 1000,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}


def get_input_fn(is_training):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    batch_size = BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE
    num_epochs = None if is_training else 1

    def input_fn():
        pipeline = Pipeline(
            filenames, is_training,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=PARAMS['model_dir'], session_config=session_config,
    save_summary_steps=500, save_checkpoints_secs=1200,
    log_step_count_steps=500
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=PARAMS, config=run_config)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=NUM_STEPS)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 2, throttle_secs=3600 * 2,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)