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

"""Tests for object_detection.matchers.argmax_matcher."""

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.matchers import atss_matcher
from object_detection.utils import test_case
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import box_list

class AtssMatcherTest(test_case.TestCase):

  @staticmethod
  def gen_anchors(level_num):
    level_x = np.linspace(0, 1, level_num)
    level_y = np.linspace(0, 1, level_num)
    level_xx, level_yy = np.meshgrid(level_x, level_y)
    level_yy_min = np.delete(level_yy, level_num-1, 0)
    level_yy_min = np.delete(level_yy_min, 0, 1)
    level_yy_max = np.delete(level_yy, 0, 0)
    level_yy_max = np.delete(level_yy_max, 0, 1)
    level_xx_min = np.delete(level_xx, level_num-1, 1)
    level_xx_min = np.delete(level_xx_min, 0, 0)
    level_xx_max = np.delete(level_xx, 0, 1)
    level_xx_max = np.delete(level_xx_max, 0, 0)
    level_anchors = np.stack([level_yy_min.ravel(), level_xx_min.ravel(), level_yy_max.ravel(), level_xx_max.ravel()], axis=-1)
    return level_anchors

  def test_return_correct_matches_single_match(self):

    def graph_fn(similarity_matrix, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims):
      anchor_level_indices = tf.unstack(tf.cast(anchor_level_indices, tf.int32), axis=0)
      feature_map_spatial_dims = tf.unstack(tf.cast(feature_map_spatial_dims, tf.int32), axis=0)
      gt_boxes = box_list.BoxList(tf.cast(gt_boxes, tf.float32))
      anchors = box_list.BoxList(tf.cast(anchors, tf.float32))
      matcher = atss_matcher.AtssMatcher(use_matmul_gather=True, number_sample_per_level_per_anchor_on_loc=[2])
      match = matcher.match(similarity_matrix, gt_boxes=gt_boxes, anchors=anchors, anchor_level_indices=anchor_level_indices, feature_map_spatial_dims=feature_map_spatial_dims)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)



    def graph_similarity_cal(gt_boxes, anchors):
      gt_boxes_list = box_list.BoxList(tf.cast(gt_boxes, dtype=tf.float32))
      anchors_box_list = box_list.BoxList(tf.cast(anchors, dtype=tf.float32))
      return similarity_calc.compare(gt_boxes_list, anchors_box_list)


    similarity_calc = sim_calc.IouSimilarity()
    gt_boxes = np.array([
      [0.2, 0.2, 0.7, 0.4],
      [0.4, 0.4, 0.8, 0.8]
    ])

    level1_anchors = self.gen_anchors(4)
    level2_anchors = self.gen_anchors(3)

    anchors = np.concatenate([level1_anchors, level2_anchors], axis=0)

    anchor_level_indices = np.array([level1_anchors.shape[0], level2_anchors.shape[0]])
    feature_map_spatial_dims = np.array([[3, 3], [2, 2]])

    similarity = self.execute(graph_similarity_cal, [gt_boxes, anchors])

    (res_matched_cols, res_unmatched_cols,
     res_match_results) = self.execute(graph_fn, [similarity, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims])

    expected_matched_rows = np.array([1])
    #matched to gt 1
    self.assertAllEqual(res_match_results[res_matched_cols], expected_matched_rows)
    #no.4 anchor matched
    expected_matched_column = np.array([4])
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_column)
    self.assertFalse(np.all(res_unmatched_cols))

  def test_return_correct_matches_multiple_matches_per_anchor(self):

    def graph_fn(similarity_matrix, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims):
      anchor_level_indices = tf.unstack(tf.cast(anchor_level_indices, tf.int32), axis=0)
      feature_map_spatial_dims = tf.unstack(tf.cast(feature_map_spatial_dims, tf.int32), axis=0)
      gt_boxes = box_list.BoxList(tf.cast(gt_boxes, tf.float32))
      anchors = box_list.BoxList(tf.cast(anchors, tf.float32))
      matcher = atss_matcher.AtssMatcher(use_matmul_gather=True, number_sample_per_level_per_anchor_on_loc=[5])
      match = matcher.match(similarity_matrix, gt_boxes=gt_boxes, anchors=anchors, anchor_level_indices=anchor_level_indices, feature_map_spatial_dims=feature_map_spatial_dims)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    def graph_similarity_cal(gt_boxes, anchors):
      gt_boxes_list = box_list.BoxList(tf.cast(gt_boxes, dtype=tf.float32))
      anchors_box_list = box_list.BoxList(tf.cast(anchors, dtype=tf.float32))
      return similarity_calc.compare(gt_boxes_list, anchors_box_list)

    similarity_calc = sim_calc.IouSimilarity()
    gt_boxes = np.array([
      [0.2, 0.2, 0.7, 0.4],
      [0.4, 0.4, 0.8, 0.8],
      [0.15, 0.15, 0.9, 0.9],
      [0.15, 0.15, 0.5, 0.5]
    ])

    level1_anchors = self.gen_anchors(4)
    level2_anchors = self.gen_anchors(3)

    anchors = np.concatenate([level1_anchors, level2_anchors], axis=0)

    anchor_level_indices = np.array([level1_anchors.shape[0], level2_anchors.shape[0]])
    feature_map_spatial_dims = np.array([[3, 3], [2, 2]])
    similarity = self.execute(graph_similarity_cal, [gt_boxes, anchors])

    (res_matched_cols, res_unmatched_cols,
     res_match_results) = self.execute(graph_fn, [similarity, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims])

    expected_matched_rows = np.array([1, 3, 1])
    #matched to gt 1
    self.assertAllEqual(res_match_results[res_matched_cols], expected_matched_rows)
    #no.4 anchor matched
    expected_matched_column = np.array([4, 9, 12])
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_column)
    self.assertFalse(np.all(res_unmatched_cols))

  def test_return_correct_matches_single_match_fixed_threshold(self):

    def graph_fn(similarity_matrix, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims):
      anchor_level_indices = tf.unstack(tf.cast(anchor_level_indices, tf.int32), axis=0)
      feature_map_spatial_dims = tf.unstack(tf.cast(feature_map_spatial_dims, tf.int32), axis=0)
      gt_boxes = box_list.BoxList(tf.cast(gt_boxes, tf.float32))
      anchors = box_list.BoxList(tf.cast(anchors, tf.float32))
      matcher = atss_matcher.AtssMatcher(use_matmul_gather=True, number_sample_per_level_per_anchor_on_loc=[2], fixed_iou_threshold=0.3)
      match = matcher.match(similarity_matrix, gt_boxes=gt_boxes, anchors=anchors, anchor_level_indices=anchor_level_indices, feature_map_spatial_dims=feature_map_spatial_dims)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)



    def graph_similarity_cal(gt_boxes, anchors):
      gt_boxes_list = box_list.BoxList(tf.cast(gt_boxes, dtype=tf.float32))
      anchors_box_list = box_list.BoxList(tf.cast(anchors, dtype=tf.float32))
      return similarity_calc.compare(gt_boxes_list, anchors_box_list)


    similarity_calc = sim_calc.IouSimilarity()
    gt_boxes = np.array([
      [0.2, 0.2, 0.7, 0.4],
      [0.4, 0.4, 0.8, 0.8]
    ])

    level1_anchors = self.gen_anchors(4)
    level2_anchors = self.gen_anchors(3)

    anchors = np.concatenate([level1_anchors, level2_anchors], axis=0)

    anchor_level_indices = np.array([level1_anchors.shape[0], level2_anchors.shape[0]])
    feature_map_spatial_dims = np.array([[3, 3], [2, 2]])

    similarity = self.execute(graph_similarity_cal, [gt_boxes, anchors])

    (res_matched_cols, res_unmatched_cols,
     res_match_results) = self.execute(graph_fn, [similarity, gt_boxes, anchors, anchor_level_indices, feature_map_spatial_dims])

    expected_matched_rows = np.array([1])
    #matched to gt 1
    self.assertAllEqual(res_match_results[res_matched_cols], expected_matched_rows)
    #no.4 anchor matched
    expected_matched_column = np.array([4])
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_column)
    self.assertFalse(np.all(res_unmatched_cols))
if __name__ == '__main__':
  tf.test.main()
