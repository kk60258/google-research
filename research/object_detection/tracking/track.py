from tracker.multitracker import JDETracker, STrack
from utils.timer import Timer
from utils import visualization as vis
from utils.log import logger
from utils.evaluation import Evaluator
from detector.interpreter import Model
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import glob
import logging
import motmetrics as mm
import pathlib

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq(opt, images, result_filename, save_dir=None, show_image=False, data_type='mot', frame_rate=30):
    '''

    :param opt: argument from parse_args
    :param images:
    :param result_filename:
    :param data_type:
    :return:
    '''
    model = Model(opt)
    tracker = JDETracker(opt, frame_rate)
    timer_model = Timer('model')
    timer_tracker = Timer('tracker')
    results = []
    # images = glob.glob(os.path.join(opt.image_dir, '*.jpg'))

    for frame_id, image_path in enumerate(images):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_rgb.shape[0] != 300 or image_rgb.shape[1] != 300:
            image_rgb = cv2.resize(image_rgb, (300, 300))

        timer_model.tic()
        pred_bboxs, pred_scores, pred_embeddings = model.run(image_rgb, image.shape[0], image.shape[1])
        timer_model.toc()

        # pred_bboxs = [np.array([0.2, 0.3, 0.6, 0.4]), np.array([0.3, 0.2, 0.4, 0.9]), np.array([0.4, 0.0, 0.6, 0.3])] * 10
        # pred_scores = [0.4, 0.6, 0.9]
        # pred_embeddings = [np.arange(5.), np.arange(5., 10.), np.arange(11., 16.)]

        # pred_bboxs_tlwh = [STrack.tlbr_to_tlwh(box) for box in pred_bboxs]
        # online_im_test = vis.plot_tracking(image, pred_bboxs_tlwh, list(range(len(pred_bboxs))), frame_id=frame_id,
        #                               fps=1. / 1)
        # f = os.path.join(save_dir, '{}_.jpg'.format(os.path.basename(image_path)[:-4]))
        # print(f)
        # cv2.imwrite(f, online_im_test)

        timer_tracker.tic()
        online_targets = tracker.update(pred_bboxs, pred_scores, pred_embeddings)
        timer_tracker.toc()

        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
            print(tlwh, tid)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            average_time = (timer_model.total_time + timer_tracker.total_time + 1e-10) / timer_model.calls
            online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / average_time)
        if show_image:
            cv2.imshow('online_im', online_im)

        if save_dir is not None:
            f = os.path.join(save_dir, '{}.jpg'.format(os.path.basename(image_path)[:-4]))
            print(f)
            cv2.imwrite(f, online_im)
    # save results
    average_time = (timer_model.total_time + timer_tracker.total_time + 1e-10) / timer_model.calls
    write_results(result_filename, results, data_type)
    return frame_id, average_time, timer_model.calls


def main(opt, data_root='/data/MOT16/train', seqs=('MOT16-05',), exp_name='demo', save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(opt.saved_model_dir, '..', 'results', exp_name)
    pathlib.Path(result_root).mkdir(parents=True, exist_ok=True)
    data_type = 'mot'
    #
    # # Read config
    # cfg_dict = parse_model_cfg(opt.cfg)
    # opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(opt.saved_model_dir, '..','outputs', exp_name, seq) if save_images or save_videos else None
        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info('start seq: {}'.format(seq))
        images = sorted(glob.glob(osp.join(data_root, seq, 'img1', '*.jpg')))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, images, result_filename, data_type=data_type,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


def parse_args(args=None):
    '''

    :param args:
    :return:
    '''
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--mot_data_root', default='', type=str)
    parser.add_argument('--min_box_area', default=200, type=float, help='filter out tiny boxes')
    parser.add_argument('--show_image', default=False, type=bool)
    # tracker
    parser.add_argument('--track_buffer', default=30, type=int)
    parser.add_argument('--new_track_thres', default=0.5, type=float)
    # detector
    parser.add_argument('--score_thres', default=0.5, type=float)
    parser.add_argument('--saved_model_dir', default='', type=str)

    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')

    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    return parser.parse_args(args)

if __name__ == '__main__':

    opt = parse_args()
    print(opt, end='\n\n')

    if not opt.test_mot16:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
        # seqs_str = '''MOT17-02-SDP
        #             '''
        # data_root = '/home/wangzd/datasets/MOT/MOT17/images/train'
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        # data_root = '/home/wangzd/datasets/MOT/MOT16/images/test'
    data_root = opt.mot_data_root
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         show_image=False,
         save_images=opt.save_images,
         save_videos=opt.save_videos)