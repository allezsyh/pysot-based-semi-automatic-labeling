from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', default='../experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='../experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
#####
parser.add_argument('--video_path', default='Y:/children_hospital_dataset_2/select_frames/room_2', type=str, help='videos or image files')
parser.add_argument('--start_video', default='camera2-2021_12_29_06-00-01_1', type=str, help='videos or image files')
parser.add_argument('--end_video', default='camera2-2022_01_03_18-29-26_1', type=str, help='videos or image files')
parser.add_argument('--txt_root', default='Y:/children_hospital_dataset_2/labels', help='datasets')
parser.add_argument('--cls_id', default=0.2, help='class id')
args = parser.parse_args()
global_flag = True
args.txt_root += ('/' + str(args.cls_id))
video_ls = os.listdir(args.video_path)
video_ls.sort()
assert args.start_video in video_ls and args.end_video in video_ls and video_ls.index(args.start_video) <= video_ls.index(args.end_video)
v_ls = []
flag = False
for v in video_ls:
    if v == args.start_video:
        flag = True
    if flag:
        v_ls.append(v)
    if v == args.end_video:
        flag = False

def get_frames(one_split_path):
    images = []
    resume_split, resume_txt_id = 0, 0
    if os.path.exists((str(args.cls_id)+'_last_split.txt')):
        with open((str(args.cls_id)+'_last_split.txt'), "r") as rl:
            last = rl.readline()
        if len(last):
            resume_split = int(last.split(os.sep)[-2].split('_')[-1])
            resume_txt_id = int(last.split('_')[-1].split('.')[0])
    images_split = glob(os.path.join(one_split_path, '*.jp*'))
    for img in images_split:
        if int(img.split(os.sep)[-2].split('_')[-1]) >= resume_split:
            if int(img.split('_')[-1].split('.')[0]) >= resume_txt_id:
                images.append(img)
    for i, img in enumerate(images):
        frame_path = images[i]
        frame = cv2.imread(img)
        yield frame, frame_path


def main(global_flag):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    ward_id = args.video_name.split('/')[-2]
    ward_path = os.path.join(args.txt_root, ward_id)
    if not os.path.exists(ward_path):
        os.makedirs(ward_path)
    video_name = args.video_name.split('/')[-1]
    vn_path = os.path.join(ward_path, video_name)
    if not os.path.exists(vn_path):
        os.makedirs(vn_path)


    resume_split, resume_txt_id = 0, 0
    if os.path.exists((str(args.cls_id)+'_last_split.txt')):
        with open((str(args.cls_id)+'_last_split.txt'), "r") as rl:
            last = rl.readline()
        if len(last):
            resume_split = int(last.split(os.sep)[-2].split('_')[-1])
            resume_txt_id = int(last.split('_')[-1].split('.')[0])
    key = 0
    split_n_ls = os.listdir(args.video_name)
    split_i_ls = []
    for split_n in split_n_ls:
        split_i = int(split_n.split('_')[-1])
        split_i_ls.append(split_i)
    split_i_ls.sort()
    for one_split_id in split_i_ls:
        one_split = 'split_'+str(one_split_id)
        one_split_path = os.path.join(args.video_name, one_split)
        first_frame = True
        if one_split_id < resume_split:
            continue
        for frame, frame_path in get_frames(one_split_path):
            img_w, img_h = frame.shape[1], frame.shape[0]
            img_name = frame_path.split(os.sep)[-1].split('.')[0]
            img_name_id = img_name.split('_')[-1]
            img_name_head = img_name.split(img_name_id)[0]
            img_name_id = int(img_name_id)
            frame_path_head = frame_path.split(img_name)[0]
            split = frame_path.split(os.sep)[-2]
            split_min_id = (int(split.split('_')[-1])-1)*300 + 1
            split_path = os.path.join(vn_path, split)
            if not os.path.exists(split_path):
                os.makedirs(split_path)
            txt_name = img_name + '.txt'
            txt_path = os.path.join(split_path, txt_name)
            if first_frame:
                try:
                    cv2.namedWindow(args.video_name, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(args.video_name, 100, 100)
                    im0 = frame.copy()
                    im0 = draw_grid_line(im0, 50)
                    cv2.putText(im0, (split+'/'+txt_name), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness=2)
                    init_rect = cv2.selectROI(args.video_name, im0, False, False)
                    x, y, w, h = init_rect
                except:
                    exit()
                if not x == y == w == h == 0:
                    tracker.init(frame, init_rect)
                first_frame = False

                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                with open(txt_path, "w") as out_file:
                    log_id = args.cls_id
                    if 10 <= args.cls_id < 100:
                        log_id = args.cls_id // 10
                    if 0 < args.cls_id < 1:  log_id = int(args.cls_id)
                    out_file.write(str(log_id) + ' ' +
                                   str(target_pos[0] / img_w) + ' ' + str(target_pos[1] / img_h) + ' ' +
                                   str(target_sz[0] / img_w) + ' ' + str(target_sz[1] / img_h)
                                   )
                print(txt_path)
            else:
                if not x == y == w == h == 0:
                    outputs = tracker.track(frame)
                    start_x = outputs['bbox'][0]
                    start_y = outputs['bbox'][1]
                    w = outputs['bbox'][2]
                    h = outputs['bbox'][3]
                    end_x = outputs['bbox'][0] + outputs['bbox'][2]
                    end_y = outputs['bbox'][1] + outputs['bbox'][3]

                    target_pos = [start_x + (w/2), start_y + (h/2)]
                    target_sz = [w, h]

                start_x = (target_pos[0] - (target_sz[0] / 2))
                end_x = (target_pos[0] + (target_sz[0] / 2))
                start_y = (target_pos[1] - (target_sz[1] / 2))
                end_y = (target_pos[1] + (target_sz[1] / 2))
                p1, p2 = (int(start_x), int(start_y)), (int(end_x), int(end_y))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)

                with open(txt_path, "w") as out_file:
                    log_id = args.cls_id
                    if 10 <= args.cls_id < 100:
                        log_id = args.cls_id // 10
                    if 0 < args.cls_id < 1:  log_id = int(args.cls_id)
                    out_file.write(str(log_id)+' '+
                                   str(target_pos[0]/img_w)+' '+str(target_pos[1]/img_h)+' '+
                                   str(target_sz[0]/img_w)+' '+str(target_sz[1]/img_h)
                                   )
                print(txt_path)

                cv2.putText(frame, (split+'/'+txt_name), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness=2)
                cv2.imshow(args.video_name, frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print('手动退出时，最后一张图片标注：', txt_path)
                    if os.path.exists((str(args.cls_id)+'_last_split.txt')):
                        with open((str(args.cls_id)+'_last_split.txt'), "r") as lr:
                            last_txt = lr.readline().strip('\n')
                            if len(last_txt):
                                last_img_name_id = int(last_txt.split('_')[-1].split('.')[0])
                            else: last_img_name_id = 1
                    else:
                        last_img_name_id = 1

                    with open((str(args.cls_id)+'_last_split.txt'), "w") as lw:
                        lw.write(txt_path)

                    if not os.path.exists('human_to_do.txt'):
                        with open('human_to_do.txt', "w") as wt:
                            for fi in range(max(img_name_id - 20, last_img_name_id), img_name_id + 1):
                                if fi < split_min_id:
                                    # frame_path_head_1 = os.path.join(frame_path_head.split(split)[0], ('split_'+str(int(split.split('_')[-1])-1)))
                                    # wt_line = os.path.join(frame_path_head_1, img_name_head + str(fi).zfill(7) + '.jpg')
                                    # wt.write(wt_line + '\n')
                                    pass
                                else:
                                    wt_line = os.path.join(frame_path_head, img_name_head + str(fi).zfill(7) + '.jpg')
                                    wt.write(wt_line + '\n')
                    else:
                        with open('human_to_do.txt', "r") as rd:
                            todo_ls = rd.readlines()
                            for ls_i in range(len(todo_ls)):
                                todo_ls[ls_i] = todo_ls[ls_i].strip('\n')
                        with open('human_to_do.txt', "a") as wa:
                            for fi in range(max(img_name_id - 20, last_img_name_id), img_name_id + 1):
                                if fi < split_min_id:
                                    # frame_path_head_1 = os.path.join(frame_path_head.split(split)[0], ('split_' + str(int(split.split('_')[-1]) - 1)))
                                    # wa_line = os.path.join(frame_path_head_1, img_name_head + str(fi).zfill(7) + '.jpg')
                                    # if wa_line not in todo_ls:
                                    #     wa.write(wa_line + '\n')
                                    pass
                                else:
                                    wa_line = os.path.join(frame_path_head, img_name_head + str(fi).zfill(7) + '.jpg')
                                    if wa_line not in todo_ls:
                                        wa.write(wa_line + '\n')
                    break
        cv2.destroyAllWindows()
        if key == 27:  # ESC
            break
    if one_split_id == split_i_ls[-1] and key != 27:
        global_flag = False
    return global_flag

def human_plot():
    cls_id = args.cls_id
    txt_root = args.txt_root
    resume = 0

    with open('human_to_do.txt', "r") as rd:
        todo_ls = rd.readlines()
        for ls_i in range(len(todo_ls)):
            todo_ls[ls_i] = todo_ls[ls_i].strip('\n')

    img_files = todo_ls
    ims = [cv2.imread(imf) for imf in img_files]
    for i, im in enumerate(ims):
        if i < resume:
            continue
        img_w, img_h = im.shape[1], im.shape[0]
        cv2.namedWindow('human plot', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('human plot', 100, 100)
        cv2.putText(ims[i], (img_files[i].split(os.sep)[-2] + '/' + img_files[i].split(os.sep)[-1]), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), thickness=2)
        ims[i] = draw_grid_line(ims[i], 50)
        init_rect = cv2.selectROI('human plot', ims[i], False, False)
        x, y, w, h = init_rect
        # assert not x == y == w == h == 0
        # cv2.destroyAllWindows()
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])

        ward_id = img_files[i].split(os.sep)[-3].split('/')[-2]
        ward_path = os.path.join(txt_root, ward_id)
        video_name = img_files[i].split(os.sep)[-3].split('/')[-1]
        vn_path = os.path.join(ward_path, video_name)
        if not os.path.exists(vn_path):
            os.makedirs(vn_path)
        split = img_files[i].split(os.sep)[-2]
        split_path = os.path.join(vn_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        img_name = img_files[i].split(os.sep)[-1].split('.')[0]
        txt_name = img_name + '.txt'
        txt_path = os.path.join(split_path, txt_name)
        with open(txt_path, "w") as out_file:
            log_id = cls_id
            if 10 <= cls_id < 100:
                log_id = cls_id // 10
            if 0 < args.cls_id < 1:  log_id = int(args.cls_id)
            out_file.write(str(log_id) + ' ' +
                           str(target_pos[0] / img_w) + ' ' + str(target_pos[1] / img_h) + ' ' +
                           str(target_sz[0] / img_w) + ' ' + str(target_sz[1] / img_h)
                           )

        with open('human_to_do.txt', 'w') as f:
            for j in range(i + 1, len(img_files)):
                f.write(img_files[j] + '\n')
        print(i, '/', len(img_files), txt_path)

    cv2.destroyAllWindows()

def draw_grid_line(image, sapcing=0, color=(10, 10, 10)):
    hight, width, th = image.shape
    cur_hight = 0
    cur_width = 0
    while cur_hight < hight and sapcing:
        cur_hight += sapcing
        cv2.line(image, (0, cur_hight), (width, cur_hight), color)
    while cur_width < width and sapcing:
        cur_width += sapcing
        cv2.line(image, (cur_width, 0), (cur_width, hight), color)
    return image

if __name__ == '__main__':
    for v in v_ls:
        global_flag = True
        args.video_name = args.video_path + '/' + v
        while global_flag:
            global_flag = main(global_flag)
            human_plot()
            if not global_flag:
                with open((str(args.cls_id) + '_last_split.txt'), "w") as lw:
                    pass