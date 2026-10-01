Source scene: ScanNet scene0432_01. No generative imagery is used.
01_single_frame_candidates.png: frame 0 RGB with three projected native single-view candidate extents selected from the stored NMS observer log.
02_reliable_view_topk3.png: frames 0, 25, 50; one native 3D object is projected into all three views to visualize K=3 view selection.
03_boxer_3d_candidates.png: transparent isometric rendering of three actual 3D candidate cuboids from the stored observer log.
base_candidate_generation_visual_strip.png: no-text composite for direct insertion under CuTR+CLIP / Reliable-View Top-K / Boxer labels.
The first panel is a representative native-candidate visualization from stored pipeline observations; it is not a separate raw CuTR tensor dump.
03_boxer_3d_candidates_on_rgb.png: the same three stored 3D candidate cuboids reprojected onto the real RGB frame; use the caption "3D boxes are reprojected for visualization."\nbase_candidate_generation_visual_strip_on_rgb.png: recommended three-stage strip using the real-image 3D visualization in the Boxer stage.\nEOF
05_wedetect_shared_provider_2d_to_3d.png: real WeDetect-Uni post-NMS proposals from frames 0/25/50, color-matched by their lifted world-space centers. The right panel reprojects the corresponding frame-0 lifted 3D AABBs into the real RGB frame using the ScanNet color intrinsics and inverse camera pose. Cuboid edges are reconstructed from coordinate adjacency rather than assuming a stored corner order.
Visualization color convention: all RGB panels use the true RGB channel order. Multi-view frames are globally color-normalized to frame 0 for figure consistency; geometry and detections are unchanged. Object colors are fixed as orange (left table), green (right table), and blue (central ottoman).
