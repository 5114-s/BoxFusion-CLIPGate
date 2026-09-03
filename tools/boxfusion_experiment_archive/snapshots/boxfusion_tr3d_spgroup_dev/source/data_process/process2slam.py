import os
import numpy as np
import json
import cv2
import shutil
root = '/media/lyq/temp/dataset/CA-1M'
target_dir = '/media/lyq/temp/dataset/CA-1M-slam'
dir_path = os.listdir(root)
# print("dirpath",dir_path)
number_dir = sorted([i.split('-')[-1] for i in dir_path])
print("number dir",number_dir)

#['42446540', '42897501', '42897521', '42897538', '42897545', '42897552', '42897561', '42897599', '42897647', '42897688', '42897692', '42898486', '42898521', '42898538', '42898570', '42898811', '42898849', '42898867', '42899459', '42899611', '42899617', '42899679', '42899691', '42899698', '42899712', '42899725', '42899729', '42899736', '43896260', '43896321', '43896330', '44358442', '44358451', '45260854', '45260898', '45260903', '45260920', '45261121', '45261133', '45261143', '45261179', '45261575', '45261587', '45261615', '45261631', '45662921', '45662942', '45662970', '45662981', '45663113', '45663149', '45663164', '47115452', '47115469', '47115525', '47115543', '47204552', '47204559', '47204573', '47204605', '47331068', '47331262', '47331311', '47331319', '47331651', '47331661', '47331963', '47331971', '47331988', '47332000', '47332885', '47332893', '47332915', '47333431', '47333440', '47333452', '47333898', '47333916', '47333923', '47333927', '47333934', '47334107', '47334115', '47334234', '47334239', '47334256', '47430475', '47430485', '47895341', '47895364', '47895534', '47895542', '47895552', '48018345', '48018367', '48018375', '48018382', '48018559', '48018566', '48018730', '48018737', '48018947', '48458415', '48458427', '48458481', '48458647', '48458654']


v_seqs = ['42897599', '42897647', '42897692', '42898521', '42898538', '42898570', '42898849', '42898867', '42899459', '42899611', '45261121', '45261133', '45261143', '45261615', '45261631', '45662921', '45662942', '47115525', '47115543', '47204605', '47331651', '47331971', '47332000', '47333934', '47334107', '47334234', '47895364', '47895534', '47895542', '47895552', '48018345', '48018367', '48018382', '48458481', '48458647', '48458654']
h_seqs = ['42446540', '42897501', '42897521', '42897538', '42897545', '42897552', '42897561', '42897688', '42898486', '42898811', '42899617', '42899679', '42899691', '42899698', '42899712', '42899725', '42899729', '42899736', '43896260', '43896321', '43896330', '44358442', '44358451', '45260854', '45260898', '45260903', '45260920', '45261179', '45261575', '45261587', '45662970', '45662981', '45663113', '45663149', '45663164', '47115452', '47115469', '47204552', '47204559', '47204573', '47331068', '47331262', '47331311', '47331319', '47331661', '47331963', '47331988', '47332885', '47332893', '47332915', '47333431', '47333440', '47333452', '47333898', '47333916', '47333923', '47333927', '47334115', '47334239', '47334256', '47430475', '47430485', '47895341', '48018375', '48018559', '48018566', '48018730', '48018737', '48018947', '48458415', '48458427']
complete_h_seqs = ['42446540', '42897501', '42897521', '42897552', '42897561', '42897688', '42898486', '42898811', '42899617', '42899691', '42899712', '42899729', '42899736', '43896260', '43896321', '43896330', '44358442', '44358451', '45260854', '45260898', '45260903', '45260920', '45261179', '45261575', '45261587', '45662970', '45662981', '45663113', '45663149', '45663164', '47115452', '47115469', '47204552', '47204559', '47204573', '47331068', '47331262', '47331319', '47332885', '47332893', '47332915', '47333431', '47333440', '47333452', '47333927', '47334239', '47334256', '47430485', '48018375', '48018559', '48018566', '48018737', '48458415', '48458427']
complete_v_seqs = ['42897647', '42897692', '42898538', '42898570', '42898849', '42898867', '42899459', '42899611', '45261121', '45261133', '45261143', '45261615', '45261631', '45662921', '45662942', '47115525', '47115543', '47204605', '47331971', '47332000', '47333934', '47334107', '47334234', '47895364', '47895534', '48018367', '48458481', '48458647', '48458654']


for seq in number_dir:
    os.makedirs(os.path.join(target_dir,seq),exist_ok=True)
    os.makedirs(os.path.join(target_dir,seq,"rgb"),exist_ok=True)
    os.makedirs(os.path.join(target_dir,seq,"depth"),exist_ok=True)
    second_path = os.path.join(root+"/ca1m-val-"+seq,seq)
    frames_dir = os.listdir(second_path)
    frames_dir = [i.split(".")[0] for i in frames_dir if 'world' not in i]
    idx = sorted(np.unique(frames_dir))
    count = 0
    all_poses = []
    all_K_rgb = []
    all_K_depth = []
    T_gravity = []

    first_h = 0 
    first_w = 0
    for frame_id in idx:
        img_path = os.path.join(second_path,frame_id+'.wide')
        gt_path = os.path.join(second_path,frame_id+'.gt')

        rgb = os.path.join(img_path,'image.png')
        depth = os.path.join(gt_path,'depth.png')
        pose = os.path.join(gt_path,'RT.json')
        gravity = os.path.join(img_path,'T_gravity.json')
        K_rgb = os.path.join(gt_path,"image",'K.json')
        K_depth = os.path.join(gt_path,"depth",'K.json')


        cur_d  = cv2.imread(depth,-1)

        if first_h==0 and first_w==0:
            # read size of first image
            first_h, first_w = cur_d.shape[0], cur_d.shape[1]

        with open(pose, 'r') as f:
            matrix = np.asarray(json.load(f))
            all_poses.append(matrix)
        with open(gravity, 'r') as f:
            matrix = np.asarray(json.load(f))
            T_gravity.append(matrix)

        if seq in complete_h_seqs or seq in complete_v_seqs:
            with open(K_rgb, 'r') as f:
                matrix = np.asarray(json.load(f))
                all_K_rgb.append(matrix)
            with open(K_depth, 'r') as f:
                matrix = np.asarray(json.load(f))
                all_K_depth.append(matrix)
        else:
            if seq in v_seqs:
                # 
                if cur_d.shape[0] > cur_d.shape[1]:
                    with open(K_rgb, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_rgb.append(matrix)
                    with open(K_depth, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_depth.append(matrix)
            else:
                if cur_d.shape[0] < cur_d.shape[1]:
                    with open(K_rgb, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_rgb.append(matrix)
                    with open(K_depth, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_depth.append(matrix)
        
        print(f"finishing {seq}")
        

        t_rgb = os.path.join(target_dir,seq,"rgb",str(count)+".png")
        t_depth = os.path.join(target_dir,seq,"depth",str(count)+".png")

        count+=1
    # Save the pose and gravity data
    t_pose_path = os.path.join(target_dir,seq,"all_poses.npy")
    t_gravity_path = os.path.join(target_dir,seq,"T_gravity.npy")
    K_rgb_path = os.path.join(target_dir,seq,"K_rgb.txt")
    K_depth_path = os.path.join(target_dir,seq,"K_depth.txt")

    all_K_rgb = np.array(all_K_rgb)
    all_K_depth = np.array(all_K_depth)
    all_K_rgb = np.mean(all_K_rgb, axis=0)  # [3,3]
    all_K_depth = np.mean(all_K_depth, axis=0)  # [3,3]


    np.savetxt(K_rgb_path, np.array(all_K_rgb)) #[N,4,4]
    np.savetxt(K_depth_path, np.array(all_K_depth)) #[N,3,3]





