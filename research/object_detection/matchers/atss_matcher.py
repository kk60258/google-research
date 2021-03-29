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

DEBUG = False

class AtssMatcher(matcher.Matcher):
  """Matcher based on AtssMatcher.

  https://arxiv.org/abs/1912.02424

  """

  def __init__(self,
               use_matmul_gather=False,
               number_sample_per_level_per_anchor_on_loc=[9, 9, 9, 9, 9, 9]):
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
    self._number_sample_per_level_per_anchor_on_loc = number_sample_per_level_per_anchor_on_loc if number_sample_per_level_per_anchor_on_loc else [9]


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

      print_op_list = []
      cy_gt = (gt_boxes_tensor[:, 0] + gt_boxes_tensor[:, 2]) * 0.5
      cx_gt = (gt_boxes_tensor[:, 1] + gt_boxes_tensor[:, 3]) * 0.5

      cy_anchors = (anchors_tensor[:, 0] + anchors_tensor[:, 2]) * 0.5
      cx_anchors = (anchors_tensor[:, 1] + anchors_tensor[:, 3]) * 0.5
      distance = -1 * ((tf.expand_dims(cy_anchors, 1) - tf.expand_dims(cy_gt, 0)) ** 2 + (tf.expand_dims(cx_anchors, 1) - tf.expand_dims(cx_gt, 0)) ** 2)
      print_op_list.append(tf.print("cy_gt ", cy_gt, summarize=-1))
      print_op_list.append(tf.print("cx_gt ", cx_gt, summarize=-1))
      print_op_list.append(tf.print("cy_anchors ", cy_anchors, summarize=-1))
      print_op_list.append(tf.print("cx_anchors ", cx_anchors, summarize=-1))
      print_op_list.append(tf.print("distance ", distance, summarize=-1))

      candidate_indices_list = []
      begin_index = tf.cast(0, tf.int32)
      # pick k candidates from each level by minimum distance.
      if len(anchor_level_indices) != len(self._number_sample_per_level_per_anchor_on_loc):
        number_sample_per_level_per_anchor_on_loc_list = self._number_sample_per_level_per_anchor_on_loc[0] * len(anchor_level_indices)
      else:
        number_sample_per_level_per_anchor_on_loc_list = self._number_sample_per_level_per_anchor_on_loc

      for number_anchors_per_level, spatial_size_per_level, number_sample_per_level_per_anchor_on_loc in zip(anchor_level_indices, feature_map_spatial_dims, number_sample_per_level_per_anchor_on_loc_list):
        last_index = begin_index + number_anchors_per_level
        distance_in_the_level = distance[begin_index:last_index, :]
        k = tf.cast((number_anchors_per_level / (spatial_size_per_level[0] * spatial_size_per_level[1])) * number_sample_per_level_per_anchor_on_loc, tf.int32)
        k = tf.math.minimum(tf.shape(distance_in_the_level)[0], k)  # todo k??
        print_op_list.append(tf.print("k ", k, "number_anchors_per_level ", number_anchors_per_level, summarize=-1))
        transpose = tf.transpose(distance_in_the_level, [1, 0])
        _, top_k_indices = tf.math.top_k(transpose, k, sorted=True)
        print_op_list.append(tf.print("top_k_indices ", top_k_indices + begin_index, "number_anchors_per_level ", number_anchors_per_level, summarize=-1))
        candidate_indices_list.append(begin_index + top_k_indices)
        begin_index = last_index

      # concat all picked candidate
      candidate_indices = tf.concat(candidate_indices_list, axis=1) # shape: (N, number of picks)
      candidate_indices = tf.cast(candidate_indices, dtype=tf.int32)

      # calculate iou requirement that iou > iou_mean + iou_std
      ious = similarity_matrix  # iou_calc.compare(gt_boxes, anchors)  # (N, M)

      print_op_list.append(tf.print("candidate_indices ", candidate_indices, summarize=-1))
      print_op_list.append(tf.print("ious ", ious, summarize=-1))
      # candidate_ious = self._gather_op(ious, candidate_indices)
      candidate_gathered_ious = tf.gather(ious, candidate_indices, axis=1, batch_dims=1)  # (N, M), (N, number of picks), axis=1, batch_dims=1 => N, number of picks
      print_op_list.append(tf.print("candidate_gathered_ious ", candidate_gathered_ious, summarize=-1))

      candidate_gathered_ious_mean = tf.math.reduce_mean(candidate_gathered_ious, axis=-1)
      candidate_gathered_ious_std = tf.math.sqrt(tf.math.reduce_variance(candidate_gathered_ious, axis=-1))
      candidate_gathered_ious_threshold = tf.expand_dims(candidate_gathered_ious_mean + candidate_gathered_ious_std, -1)
      candidate_gathered_iou_pass_indices = tf.greater(candidate_gathered_ious, candidate_gathered_ious_threshold)

      print_op_list.append(tf.print("candidate_gathered_ious_mean ", candidate_gathered_ious_mean, summarize=-1))
      print_op_list.append(tf.print("candidate_gathered_ious_variance ", tf.math.reduce_variance(candidate_gathered_ious, axis=-1), summarize=-1))
      print_op_list.append(tf.print("candidate_gathered_ious_std ", candidate_gathered_ious_std, summarize=-1))
      print_op_list.append(tf.print("candidate_gathered_ious_threshold ", candidate_gathered_ious_threshold, summarize=-1))
      print_op_list.append(tf.print("candidate_gathered_iou_pass_indices ", candidate_gathered_iou_pass_indices, summarize=-1))


      gt_boxes_tensor = tf.cast(gt_boxes_tensor, dtype=tf.float32)
      anchors_tensor = tf.cast(anchors_tensor, dtype=tf.float32)
      candidate_anchors = self._gather_op(anchors_tensor, candidate_indices)
      # calculate center requirement that anchor center must be inside gt_box
      gt_boxes_tensor_expanded = tf.expand_dims(gt_boxes_tensor, 1)
      candidate_anchors_center_y = (candidate_anchors[..., 0] + candidate_anchors[..., 2]) * 0.5
      candidate_anchors_center_x = (candidate_anchors[..., 1] + candidate_anchors[..., 3]) * 0.5
      # cy_anchor - y1_gt > 0 and y2_gt - cy_anchor > 0
      # cx_anchor - x1_gt > 0 and x2_gt - cx_anchor > 0
      top = candidate_anchors_center_y - gt_boxes_tensor_expanded[..., 0]
      left = candidate_anchors_center_x - gt_boxes_tensor_expanded[..., 1]
      bottom = gt_boxes_tensor_expanded[..., 2] - candidate_anchors_center_y
      right = gt_boxes_tensor_expanded[..., 3] - candidate_anchors_center_x
      diff = tf.stack([top, left, bottom, right], -1)
      candidate_center_pass_indices = tf.greater(tf.math.reduce_min(diff, -1), 0)

      print_op_list.append(tf.print("anchors_tensor ", anchors_tensor, summarize=-1))
      print_op_list.append(tf.print("candidate_anchors ", candidate_anchors, summarize=-1))
      print_op_list.append(tf.print("candidate_indices ", candidate_indices, summarize=-1))
      print_op_list.append(tf.print("candidate_anchors_center_y ", candidate_anchors_center_y, summarize=-1))
      print_op_list.append(tf.print("candidate_anchors_center_x ", candidate_anchors_center_x, summarize=-1))
      print_op_list.append(tf.print("candidate_center_pass_indices ", candidate_center_pass_indices, summarize=-1))

      #select candidates which matches iou and center requirement.
      candidate_pass = tf.logical_and(candidate_gathered_iou_pass_indices, candidate_center_pass_indices)
      candidate_indices_matrix = tf.where(candidate_pass, candidate_indices, -1 * tf.ones(tf.shape(candidate_pass), dtype=tf.int32))
      candidate_indices_one_hot = tf.one_hot(candidate_indices_matrix, depth=tf.shape(anchors_tensor)[0])  # shape: (N, number of picks, M)
      candidate_indice_matrix_binary = tf.cast(tf.math.reduce_max(candidate_indices_one_hot, 1), tf.float32)  # shape: (N, M)

      print_op_list.append(tf.print("anchors_tensor ", anchors_tensor, summarize=-1))
      print_op_list.append(tf.print("candidate_anchors ", candidate_anchors, summarize=-1))
      print_op_list.append(tf.print("candidate_indices ", candidate_indices, summarize=-1))
      print_op_list.append(tf.print("candidate_gathered_iou_pass_indices ", candidate_gathered_iou_pass_indices, summarize=-1))
      print_op_list.append(tf.print("candidate_center_pass_indices ", candidate_center_pass_indices, summarize=-1))
      print_op_list.append(tf.print("candidate_pass ", candidate_pass, summarize=-1))
      print_op_list.append(tf.print("candidate_indices_matrix ", candidate_indices_matrix, summarize=-1))
      print_op_list.append(tf.print("candidate_indices_one_hot ", candidate_indices_one_hot, summarize=-1))
      print_op_list.append(tf.print("candidate_indice_matrix_binary ", candidate_indice_matrix_binary, summarize=-1))

      # if an anchor is matched to multiple gt_box, keep the one with largest iou.
      iou_weighted_candidate_pick = candidate_indice_matrix_binary * ious  # shape: (N, M)
      matches = tf.argmax(iou_weighted_candidate_pick, axis=0, output_type=tf.int32)  # shape: (M)

      # add negative matches
      negative_indicator = tf.equal(matches, 0)
      final_matches = self._set_values_using_indicator(matches, negative_indicator, -1)

      print_op_list.append(tf.print("ious ", ious, summarize=-1))
      print_op_list.append(tf.print("candidate_indice_matrix_binary ", candidate_indice_matrix_binary, summarize=-1))
      print_op_list.append(tf.print("iou_weighted_candidate_pick ", iou_weighted_candidate_pick, summarize=-1))

      print_op_list.append(tf.print("matches ", matches, summarize=-1))
      print_op_list.append(tf.print("negative_indicator ", negative_indicator, summarize=-1))
      print_op_list.append(tf.print("final_matches ", final_matches, summarize=-1))

      if DEBUG:
        self.summarize(candidate_gathered_ious=candidate_gathered_ious,
                       candidate_gathered_ious_mean=candidate_gathered_ious_mean,
                       candidate_gathered_ious_std=candidate_gathered_ious_std,
                       candidate_gathered_iou_pass_indices=candidate_gathered_iou_pass_indices,
                       candidate_center_pass_indices=candidate_center_pass_indices,
                       candidate_pass=candidate_pass)

        with tf.control_dependencies(print_op_list):
          final_matches = final_matches * 1

      return final_matches

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

  def summarize(self, candidate_gathered_ious, candidate_gathered_ious_mean, candidate_gathered_ious_std,
                candidate_gathered_iou_pass_indices, candidate_center_pass_indices, candidate_pass):

    def summarize_batch_avg(value, name):
      avg_value = tf.reduce_mean(tf.cast(value, dtype=tf.float32))
      tf.summary.scalar('AtssMatcher/{}'.format(name), avg_value, family='TargetAssignment')

    def summarize_max_std_sample(iou_value, mean_value, std_value):
      iou_value = tf.cast(iou_value, dtype=tf.float32)
      mean_value= tf.cast(mean_value, dtype=tf.float32)
      std_value = tf.cast(std_value, dtype=tf.float32)

      max_std_index = tf.argmax(std_value, axis=0)
      iou_value_gathered = iou_value  # tf.gather(iou_value, max_std_index)
      iou_value_gathered_min = tf.reduce_min(iou_value_gathered)
      iou_value_greater_0 = tf.greater(iou_value_gathered, 0.01)
      iou_value_greater_0_count = tf.reduce_sum(tf.cast(iou_value_greater_0, tf.int32))
      mean_value_gathered = tf.gather(mean_value, max_std_index)
      std_value_gathered = tf.gather(std_value, max_std_index)
      tf.summary.scalar('AtssMatcher/{}'.format('gt_iou_min'), iou_value_gathered_min, family='TargetAssignment')
      tf.summary.scalar('AtssMatcher/{}'.format('gt_iou_greater_0_count'), iou_value_greater_0_count, family='TargetAssignment')
      tf.summary.scalar('AtssMatcher/{}'.format('max_std_gt_iou_mean'), mean_value_gathered, family='TargetAssignment')
      tf.summary.scalar('AtssMatcher/{}'.format('max_std_gt_iou_std'), std_value_gathered, family='TargetAssignment')

    shape = shape_utils.combined_static_and_dynamic_shape(candidate_gathered_ious)
    gathered_count = shape[1]
    gt_count = shape[0]
    ## if gt box is padded to max_count (100) (unpad_groundtruth_tensors: false), the following avg value will be wrong.
    tf.summary.scalar('AtssMatcher/{}'.format('GatheredIOUCount'), gathered_count, family='TargetAssignment')
    tf.summary.scalar('AtssMatcher/{}'.format('GTCount'), gt_count, family='TargetAssignment')

    summarize_batch_avg(candidate_gathered_ious_mean, 'GatheredIOUMeanAvgOnGT')
    summarize_batch_avg(candidate_gathered_ious_std, 'GatheredIOUStdAvgOnGT')

    candidate_gathered_iou_pass_count = tf.reduce_sum(tf.cast(candidate_gathered_iou_pass_indices, tf.int32), axis=-1)
    summarize_batch_avg(candidate_gathered_iou_pass_count, 'GatheredIOUPassAvgOnGT')

    candidate_center_pass_count = tf.reduce_sum(tf.cast(candidate_center_pass_indices, tf.int32), axis=-1)
    summarize_batch_avg(candidate_center_pass_count, 'GatheredCenterPassAvgOnGT')

    candidate_pass_count = tf.reduce_sum(tf.cast(candidate_pass, tf.int32), axis=-1)
    summarize_batch_avg(candidate_pass_count, 'GatheredPassAvgOnGT')

    summarize_max_std_sample(candidate_gathered_ious, candidate_gathered_ious_mean, candidate_gathered_ious_std)



