
## Data Preparation for CA-1M

Before this step, make sure you download all the validations `tar`. Remember to change the data root path.

### step1: unzip
please run the shell to unzip all the `***.tar` you have downloaded.
```
cd data_process
bash ca1m_unzip.bash
```

### step2: check images

Since there are some sequences that have images (H>W) and images (W>H) at the same time, we need to rotate them for consistency to keep all the images in one sequence are H>W or W>H. Run this command to get the information
```
python check_img.py
```

### step3: rotate images

Run this command to rotate the images
```
python rot_img.py
```

### step4: Extract images

Run this command to extract the images and other information from the original folders.
```
python process2slam.py
```

### step5: Extract GT boxes

Run this command to extract the GT boxes.
```
python process2slam_gtbox.py
```

### step6: Filter GT boxes

Run this command to filter the GT boxes, since the original GT boxes are beyond the scanning environment. There are many boxes that are useless and need to be cropped.
```
python filter_gt_boxes.py
```

As for the `mesh.ply`, you can use `open3d` to run the reconstruction based on the RGB and depth images.