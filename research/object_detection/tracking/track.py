from tracker.multitracker import JDETracker
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

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str)
parser.add_argument('--min_box_area', default=0.4, type=float)
parser.add_argument('--show_image', default=False, type=bool)
parser.add_argument('--save_dir', default='', type=str)
# tracker
parser.add_argument('--track_buffer', default=30, type=int)
parser.add_argument('--conf_thres', default=0.4, type=float)
# detector
parser.add_argument('--det_thres', default=0.3, type=float)
parser.add_argument('--saved_model_dir', default='', type=str)


opt = parser.parse_args()

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

def eval_seq(result_filename, images, data_type):
    model = Model(opt)
    tracker = JDETracker(opt)
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
        pred_bboxs, pred_scores, pred_embeddings = model.run(image_rgb)
        timer_model.toc()

        # pred_bboxs = [np.array([0.2, 0.3, 0.6, 0.4]), np.array([0.3, 0.2, 0.4, 0.9]), np.array([0.4, 0.0, 0.6, 0.3])] * 10
        # pred_scores = [0.4, 0.6, 0.9]
        # pred_embeddings = [np.arange(5.), np.arange(5., 10.), np.arange(11., 16.)]
        timer_tracker.tic()
        online_targets = tracker.update(pred_bboxs, pred_scores, pred_embeddings)
        timer_tracker.toc()

        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            print(tlwh, tid)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if opt.show_image or opt.save_dir is not None:
            average_time = (timer_model.total_time + timer_tracker.total_time + 1e-10) / timer_model.calls
            online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / average_time)
        if opt.show_image:
            cv2.imshow('online_im', online_im)

        if opt.save_dir is not None:
            cv2.imwrite(os.path.join(opt.save_dir, '{}.jpg'.format(os.path.basename(image_path)[:-4])), online_im)
    # save results
    average_time = (timer_model.total_time + timer_tracker.total_time + 1e-10) / timer_model.calls
    write_results(result_filename, results, 'mot')
    return frame_id, average_time, timer_model.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    # mkdir_if_missing(result_root)
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
        output_dir = os.path.join(data_root, '..','outputs', exp_name, seq) if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        images = glob.glob(osp.join(data_root, seq, 'img1', '*.jpg'))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, images, data_type, result_filename,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    opt = parser.parse_args()
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
        data_root = '/home/wangzd/datasets/MOT/MOT17/images/train'
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        data_root = '/home/wangzd/datasets/MOT/MOT16/images/test'
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-2],
         show_image=False,
         save_images=opt.save_images,
         save_videos=opt.save_videos)