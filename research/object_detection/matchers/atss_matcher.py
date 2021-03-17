# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Argmax matcher implementation.

This class takes a similarity matrix and matches columns to rows based on the
maximum value per column. One can specify matched_thresholds and
to prevent columns from matching to rows (generally resulting in a negative
training example) and unmatched_theshold to ignore the match (generally
resulting in neither a positive or negative training example).

This matcher is used in Fast(er)-RCNN.

Note: matchers are used in TargetAssigners. There is a create_target_assigner
factory function for popular implementations.
"""
import tensorflow.compat.v1 as tf

from object_detection.core import matcher
from object_detection.utils import shape_utils
from object_detection.utils import ops
from object_detection.core import box_list

class AtssMatcher(matcher.Matcher):
  """Matcher based on AtssMatcher.

  https://arxiv.org/abs/1912.02424

  """

  def __init__(self,
               use_matmul_gather=False,
               number_sample_per_level_per_anchor_on_loc=9):
    """Construct AtssMatcher.

    Args:
      use_matmul_gather: Force constructed match objects to use matrix
        multiplication based gather instead of standard tf.gather.
        (Default: False).

    Raises:
      ValueError: if unmatched_threshold is set but matched_threshold is not set
        or if unmatched_threshold > matched_threshold.
    """
    super(AtssMatcher, self).__init__(use_matmul_gather=use_matmul_gather)

    ## copy from matcher.Match to have the same behaviour as other matcher enherent class
    self._gather_op = tf.gather
    if use_matmul_gather:
        self._gather_op = ops.matmul_gather_on_zeroth_axis
    self._number_sample_per_level_per_anchor_on_loc = number_sample_per_level_per_anchor_on_loc


  def _match(self, similarity_matrix, valid_rows, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims, **kwargs):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.
      valid_rows: a boolean tensor of shape [N] indicating valid rows.

    Returns:
      Match object with corresponding matches for each of M columns.
    """
    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
          similarity_matrix)
      return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """

      # atss calculate the center distance between M anchors and N gt_boxes
      gt_boxes_tensor = gt_boxes.get()
      anchors_tensor = anchors.get()

      cy_gt = (gt_boxes_tensor[:, 0] + gt_boxes_tensor[:, 2]) * 0.5
      cx_gt = (gt_boxes_tensor[:, 1] + gt_boxes_tensor[:, 3]) * 0.5

      cy_anchors = (anchors_tensor[:, 0] + anchors_tensor[:, 2]) * 0.5
      cx_anchors = (anchors_tensor[:, 1] + anchors_tensor[:, 3]) * 0.5
      distance = -1 * ((tf.expand_dims(cy_anchors, 1) - tf.expand_dims(cy_gt, 0)) ** 2 + (tf.expand_dims(cx_anchors, 1) - tf.expand_dims(cx_gt, 0)) ** 2)

      candidate_indices_list = []
      begin_index = 0

      # pick k candidates from each level by minimum distance.
      for number_anchors_per_level, spatial_size_per_level in zip(anchor_level_indices, feature_map_spatial_dims):
        last_index = begin_index + number_anchors_per_level
        distance_in_the_level = distance[begin_index:last_index, :]
        k = int(number_anchors_per_level / (spatial_size_per_level[0] * spatial_size_per_level[1])) * self._number_sample_per_level_per_anchor_on_loc
        k = tf.math.minimum(tf.shape(distance_in_the_level)[0], k)  # todo k??
        transpose = tf.transpose(distance_in_the_level, [1, 0])
        _, top_k_indices = tf.math.top_k(transpose, k, sorted=False)
        candidate_indices_list.append(begin_index + top_k_indices)
        begin_index = last_index

      # concat all picked candidate
      candidate_indices = tf.concat(candidate_indices_list, axis=1) # shape: (N, number of picks)

      # calculate iou requirement that iou > iou_mean + iou_std
      ious = similarity_matrix  # iou_calc.compare(gt_boxes, anchors)  # (N, M)
      # candidate_ious = self._gather_op(ious, candidate_indices)
      candidate_ious = tf.gather(ious, candidate_indices, axis=1, batch_dims=1)  # (N, M), (N, number of picks), axis=1, batch_dims=1 => N, number of picks
      candidate_ious_mean = tf.math.reduce_mean(candidate_ious, axis=-1)
      candidate_ious_std = tf.math.sqrt(tf.math.reduce_variance(candidate_ious, axis=-1))
      candidate_ious_threshold = tf.expand_dims(candidate_ious_mean + candidate_ious_std, -1)
      candidate_iou_pass_indices = tf.greater(candidate_ious, candidate_ious_threshold)

      candidate_anchors = self._gather_op(anchors_tensor, candidate_indices)
      # calculate center requirement that anchor center must be inside gt_box
      gt_boxes_tensor_expanded = tf.expand_dims(gt_boxes_tensor, 1)
      top = gt_boxes_tensor_expanded[..., 0] - candidate_anchors[..., 0]
      left = gt_boxes_tensor_expanded[..., 1] - candidate_anchors[..., 1]
      bottom = candidate_anchors[..., 2] - gt_boxes_tensor_expanded[..., 2]
      right = candidate_anchors[..., 3] - gt_boxes_tensor_expanded[..., 3]
      diff = tf.stack([top, left, bottom, right], -1)
      candidate_center_pass_indices = tf.greater(tf.math.reduce_min(diff, -1), 0)

      #select candidates which matches iou and center requirement.
      candidate_pass = tf.logical_and(candidate_iou_pass_indices, candidate_center_pass_indices)
      candidate_indices = tf.where(candidate_pass, candidate_indices, -1 * tf.ones(tf.shape(candidate_pass), dtype=tf.int32))
      candidate_indices_one_hot = tf.one_hot(candidate_indices, depth=tf.shape(anchors_tensor)[0])  # shape: (N, number of picks, M)
      candidate_indice_matrix = tf.cast(tf.math.reduce_max(candidate_indices_one_hot, 1), tf.float32)  # shape: (N, M)

      # if an anchor is matched to multiple gt_box, keep the one with largest iou.
      iou_weighted_candidate_pick = candidate_indice_matrix * ious  # shape: (N, M)
      matches = tf.argmax(iou_weighted_candidate_pick, axis=0, output_type=tf.int32)  # shape: (M)
      return matches

    if similarity_matrix.shape.is_fully_defined():
      if shape_utils.get_dim_as_int(similarity_matrix.shape[0]) == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:
      return tf.cond(
          tf.greater(tf.shape(similarity_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)

  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)
