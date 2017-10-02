import tensorflow as tf
from tensorflow.python.framework import ops
import os

module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(module_path)
roi_pooling_module = tf.load_op_library(os.path.join(module_dir, 'roi_pooling.so'))
roi_pooling3d_module = tf.load_op_library(os.path.join(module_dir, 'roi_pooling_3d.so'))

def roi_pooling(input, rois, pool_height, pool_width):
    """
      returns a tensorflow operation for computing the Region of Interest Pooling
    
      @arg input: feature maps on which to perform the pooling operation
      @arg rois: list of regions of interest in the format (feature map index, upper left, bottom right)
      @arg pool_width: size of the pooling sections
    """
    # TODO(maciek): ops scope
    out = roi_pooling_module.roi_pooling(input, rois, pool_height=pool_height, pool_width=pool_width)
    output, argmax_output = out[0], out[1]
    return output

def roi_pooling_3d(input, rois, pool_height, pool_width, pool_depth):
    """
      returns a tensorflow operation for computing the Region of Interest Pooling
    
      @arg input: feature maps on which to perform the pooling operation
      @arg rois: list of regions of interest in the format (feature map index, upper left, bottom right)
      @arg pool_width: size of the pooling sections
    """
    # TODO(maciek): ops scope
    out = roi_pooling3d_module.roi_pooling3d(input, rois, pool_height=pool_height, pool_width=pool_width, pool_depth=pool_depth)
    output, argmax_output = out[0], out[1]
    return output


@ops.RegisterGradient("RoiPooling")
def _RoiPoolingGrad(op, *grads):
    orig_inputs = op.inputs[0]
    orig_rois = op.inputs[1]
    orig_output = op.outputs[0]
    orig_argmax_output = op.outputs[1]

    orig_output_grad = grads[0]
    output_grad = roi_pooling_module.roi_pooling_grad(orig_inputs, orig_rois, orig_output,
                                                      orig_argmax_output, orig_output_grad,
                                                      pool_height=op.get_attr('pool_height'),
                                                      pool_width=op.get_attr('pool_width'))
    return [output_grad, None]

@ops.RegisterGradient("RoiPooling3D")
def _RoiPooling3DGrad(op, *grads):
    orig_inputs = op.inputs[0]
    orig_rois = op.inputs[1]
    orig_output = op.outputs[0]
    orig_argmax_output = op.outputs[1]

    orig_output_grad = grads[0]
    output_grad = roi_pooling3d_module.roi_pooling3d_grad(orig_inputs, orig_rois, orig_output,
                                                      orig_argmax_output, orig_output_grad,
                                                      pool_height=op.get_attr('pool_height'),
                                                      pool_width=op.get_attr('pool_width'),
                                                      pool_depth=op.get_attr('pool_depth'))
    return [output_grad, None]


@ops.RegisterShape("RoiPooling")
def _RoiPoolingShape(op):
    input = op.inputs[0]
    rois = op.inputs[1]

    n_rois = rois.get_shape()[0]
    n_channels = input.get_shape()[3]
    pool_height = op.get_attr('pool_height')
    pool_width = op.get_attr('pool_width')

    #TODO: check the width/hegiht order
    return [tf.TensorShape([n_rois, n_channels, pool_width, pool_height]),
            tf.TensorShape(None)]

@ops.RegisterShape("RoiPooling3D")
def _RoiPooling3DShape(op):
    input = op.inputs[0]
    rois = op.inputs[1]

    n_rois = rois.get_shape()[0]
    n_channels = input.get_shape()[3]
    pool_height = op.get_attr('pool_height')
    pool_width = op.get_attr('pool_width')
    pool_depth = op.get_attr('pool_depth')

    #TODO: check the width/hegiht order
    return [tf.TensorShape([n_rois, n_channels, pool_width, pool_height, pool_depth]),
            tf.TensorShape(None)]
