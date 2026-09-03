import os
import numpy as np
import json
import cv2
import shutil
target_dir = '/media/lyq/temp/dataset/CA-1M-slam'
dir_path = sorted(os.listdir(target_dir))
print("dirpath",dir_path)
# v_seqs = [
#     "42897647", "42897692", "42898538", "42898570", "42898849",
#     "42898867", "42899459", "42899611", "45261121", "45261133",
#     "45261143", "45261615", "45261631", "45662921", "45662942",
#     "47115525", "47115543", "47204605", "47331651", "47331963",
#     "47331971", "47332000", "47333934", "47334107", "47334234",
#     "47895364", "47895534", "47895542", "47895552", "48018367",
#     "48018382", "48018947", "48458481", "48458647", "48458654"
# ]

h_seqs = []
v_seqs = []
complete_h_seqs = []
complete_v_seqs = []


for seq in dir_path:

    rgb_dir = os.path.join(target_dir,seq,"rgb")
    num_images = len(os.listdir(rgb_dir))
    first_h = 0 
    first_w = 0
    h_count = 0 
    v_count = 0
    for frame_id in range(num_images):

        t_rgb = os.path.join(target_dir,seq,"rgb",str(frame_id)+".png")
        t_depth = os.path.join(target_dir,seq,"depth",str(frame_id)+".png")
        
        cur_d = cv2.imread(t_depth,-1)

        if cur_d.shape[0] > cur_d.shape[1]:
            v_count+=1
        else:
            h_count+=1

    if v_count > h_count:
        v_seqs.append(seq)
        if h_count == 0:
            complete_v_seqs.append(seq)
        print(f"Vertical sequence: {seq} - Vertical image count: {v_count}, Horizontal image count: {h_count}")
    else:
        h_seqs.append(seq)
        if v_count == 0:
            complete_h_seqs.append(seq)
        print(f"Horizontal sequence: {seq} - Horizontal image count: {h_count}, Vertical image count: {v_count}")

print("v_seqs",v_seqs)
print("h_seqs",h_seqs)
print("complete_h_seqs",complete_h_seqs)
print("complete_v_seqs",complete_v_seqs)

