import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import numpy as np
import pickle
from ml import Movenet
import pickle
# import random
movenet = Movenet('movenet_thunder')

RANGE=9
videos = ["video_0001","video_0002","video_0003","video_0004","video_0005","video_0006","video_0007","video_0008","video_0009"]

def get_ped_length():
    length = 0
    for video in videos:
        _,root = paths_parser(video)
        for track in root[2:]:
            if track.attrib.get('label') == "pedestrian":
                length+=1
    return length

def get_ped_data(ped_data):
    frame = ped_data['frame']
    frame = frame.zfill(5)

    xtl = int(float(ped_data['xtl']))
    ytl = int(float(ped_data['ytl']))
    xbr = int(float(ped_data['xbr']))
    ybr = int(float(ped_data['ybr']))
    
    return frame, xtl, ytl, xbr, ybr

def paths_parser(vid):
    images_path ='images/set06/' + vid + '/'
    annotation_path= 'annotations/set06/'+ vid+ '_annt.xml'
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    return images_path, root

def bounding_box(frame, xtl, ytl, xbr, ybr):
    color = (0, 255, 0)
    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
    return frame

def get_frame(path,number):

    frame_path = os.path.join(path, f'{number}.png')
    frame = cv2.imread(frame_path)
    return frame

def get_ped_frame(frame, xtl, ytl, xbr, ybr):
    ped_frame = frame[ytl:ybr, xtl:xbr]
    if ped_frame.size == 0:
        return None
    ped_frame = cv2.resize(ped_frame, (224, 224))
    return ped_frame


def get_data(chosen_frames, element,crossing=False):
    frames = []
    ped_frames = []
    bboxs = []
    for i in chosen_frames:
        frameN, xtl, ytl, xbr, ybr = get_ped_data(element[i].attrib)
        frame = get_frame(images_path,frameN)
        if frame is None:
            return [],[],[]
        ped_frames.append(get_ped_frame(frame, xtl, ytl, xbr, ybr))
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        bboxs.append([ xtl, ytl, xbr, ybr])

    return frames, ped_frames,bboxs

def detect(ped_frames, inference_count=3):
  ped_pose = []
  for input_tensor in ped_frames:
    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
      person = movenet.detect(input_tensor,
                              reset_crop_region=False)
    ped_pose.append(person)
  return ped_pose

length = get_ped_length()
current_ped = 0
all_frames = np.empty((length, 8, 224, 224, 3)).astype(np.uint8)
all_ped_frames = np.empty((length, 8, 224, 224, 3)).astype(np.uint8)
all_bboxs = np.empty((length, 8, 4))
pose = []
labels = [None] * length
print(length)



for video in videos:
    images_path, root = paths_parser(video)
    for element in root[2:]:
        if element.attrib.get('label') == "pedestrian":
            intent = {}
            frames = []
            boxes = []
            ped_frames = []
            # get a list of crossing/not crossing timestamps
            frame_counter = 0
            for box in element:
                for item in box:
                    if item.attrib.get('name') == "cross":
                        intent[frame_counter] = item.text == "crossing"
                        break
                frame_counter+=1
            
            chosen_frames = [None] *8
            if not any(intent.values()):
            # if not crossing at all choose 8 and move on
                frames = list(intent.keys())
                middle=len(intent.keys())//2
                chosen_frames = frames[middle:middle+8]
            else:
                # if crossing, choose 8 frames around the start of the crossing event
                frames = list(intent.keys())
                first_cross = 0
                for f in frames:
                    if intent[f]:
                        first_cross = f
                        break
                chosen_frames = frames[f-23:f-15]

            frames, ped_frames, bboxs = get_data(chosen_frames,element)
            curr_pose = detect(ped_frames)
            if len(frames) == 0:
                print("skipping")
                continue
            all_frames[current_ped] = frames
            all_ped_frames[current_ped] = ped_frames
            all_bboxs[current_ped] = bboxs
            labels[current_ped] = any(intent.values())
            pose.append(curr_pose)
            print(str(current_ped)+"/"+str(length))
            current_ped+=1


data = {
    'frames': all_frames,
    'ped_frames': all_ped_frames,
    'labels': labels,
    'bounding_boxes': all_bboxs,
    'ped_pose': pose
}
    
with open('1s_test.pickle', 'wb') as handle:
    pickle.dump(data, handle)
