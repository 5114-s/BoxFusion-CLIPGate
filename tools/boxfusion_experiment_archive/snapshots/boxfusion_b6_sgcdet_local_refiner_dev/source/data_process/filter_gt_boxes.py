import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import os
import json

def filter_3d_corners(corners, K, poses, depth_maps, gt_ply_path, 
                      near=0.1, far=100.0, dist_threshold=0.3):

    print("before frustum culling", corners.shape)
    bbox_frustum_mask = frustum_culling_bbox_level(corners, K, poses, depth_maps, near, far)
    visible_bboxes = corners[bbox_frustum_mask]  # [B1, 8, 3]



    gt_points = load_gt_point_cloud(gt_ply_path)
    bbox_proximity_mask = check_bbox_proximity(visible_bboxes, gt_points, dist_threshold)
    

    filtered_corners = visible_bboxes[bbox_proximity_mask]  # [B, 8, 3]

    return filtered_corners, bbox_frustum_mask

def frustum_culling_bbox_level(corners, K, poses, depth_maps, near, far):
    """
    Frustum culling at bbox level: preserve the entire bounding box if at least one corner is inside the frustum
    """
    N = corners.shape[0]
    M = len(poses)
    bbox_mask = np.zeros((N,8), dtype=bool)  
    

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for i in range(M):
        pose = poses[i]
        depth_map = depth_maps[i]
        H, W = depth_map.shape
        pose_inv = np.linalg.inv(pose)
        # Transform from world coordinates to camera coordinates
        hom_corners = np.concatenate([corners, np.ones((N, 8, 1))], axis=-1)
        cam_points = np.dot(hom_corners, pose_inv.T)
        # cam_points = np.einsum('ijk,lk->ijl', hom_corners, pose.T)[..., :3] #[N,8,4] @ [4,4]
        
       
        x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
        
  
        valid_z = (z > near) & (z < far)
        
  
        u = (fx * x / z + cx).astype(int)
        v = (fy * y / z + cy).astype(int)
        
        
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
       
        valid_corner = valid_z & valid_uv
        
        
        bbox_mask |= valid_corner


    count_visible = np.sum(bbox_mask, axis=1)  
    bbox_visible = count_visible >= 6 

    return bbox_visible

def load_gt_point_cloud(ply_path):
    """load gt point cloud from ply file"""
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def check_bbox_proximity(bboxes, gt_points, threshold=0.3):
    """
    Bounding box level proximity check: preserve if at least one corner has GT points nearby
    """
    tree = KDTree(gt_points)
    proximity_mask = np.zeros(len(bboxes), dtype=bool)
    
    for i, bbox in enumerate(bboxes):

        dists, _ = tree.query(bbox, k=1)

        near_points_count = np.sum(dists < threshold)
        

        if near_points_count >= 4:
            proximity_mask[i] = True


    return proximity_mask


"""
Filter 3D corner points: First perform frustum culling, then verify with point cloud proximity

Parameters:
    corners: [N, 8, 3] 3D corner points in world coordinate system
    K: [3, 3] Camera intrinsic matrix
    poses: [M, 4, 4] Camera pose matrices (world to camera)
    depth_maps: [M, H, W] Array of depth maps
    gt_ply_path: Path to GT point cloud PLY file
    near: Near clipping plane distance (default 0.1m)
    far: Far clipping plane distance (default 100.0m)
    dist_threshold: Proximity distance threshold (default 0.3m)

Returns:
    filtered_corners: [K, 3] Filtered corner coordinates
    mask: [N, 8] Boolean mask indicating preserved corner positions
"""
data_root = '/media/lyq/temp/dataset/CA-1M-slam/'
all_seq = ['42446540', '42897501', '42897521', '42897538', '42897545', '42897552', '42897561', '42897599', '42897647', '42897688', '42897692', '42898486', '42898521', '42898538', '42898570', '42898811', '42898849', '42898867', '42899459', '42899611', '42899617', '42899679', '42899691', '42899698', '42899712', '42899725', '42899729', '42899736', '43896260', '43896321', '43896330', '44358442', '44358451', '45260854', '45260898', '45260903', '45260920', '45261121', '45261133', '45261143', '45261179', '45261575', '45261587', '45261615', '45261631', '45662921', '45662942', '45662970', '45662981', '45663113', '45663149', '45663164', '47115452', '47115469', '47115525', '47115543', '47204552', '47204559', '47204573', '47204605', '47331068', '47331262', '47331311', '47331319', '47331651', '47331661', '47331963', '47331971', '47331988', '47332000', '47332885', '47332893', '47332915', '47333431', '47333440', '47333452', '47333898', '47333916', '47333923', '47333927', '47333934', '47334107', '47334115', '47334234', '47334239', '47334256', '47430475', '47430485', '47895341', '47895364', '47895534', '47895542', '47895552', '48018345', '48018367', '48018375', '48018382', '48018559', '48018566', '48018730', '48018737', '48018947', '48458415', '48458427', '48458481', '48458647', '48458654']

from tqdm import tqdm  
for idx in tqdm(range(len(all_seq))):
    cur_seq = all_seq[idx]

    gt_json = os.path.join(data_root,cur_seq,'instances.json')
    with open(gt_json) as f:
        data = json.load(f)

    corners_list = [np.array(item["corners"]) for item in data]
    #  [N, 8, 3] numpy array
    corners_array = np.stack(corners_list, axis=0)

    K = np.loadtxt(os.path.join(data_root,cur_seq,'K_depth.txt')).reshape(3,3)
    poses = np.load(os.path.join(data_root,cur_seq,'all_poses.npy'))  # [M, 4, 4]
    gt_ply = os.path.join(data_root,cur_seq,'mesh.ply')
    depth_path = os.path.join(data_root,cur_seq,'depth')
    #read depth
    depth = []
    for i in range(len(os.listdir(depth_path))):
        depth_file = os.path.join(depth_path, str(i) + '.png')
        d = o3d.io.read_image(depth_file)
        depth.append(np.asarray(d))
    depth_maps = np.stack(depth, axis=0)  # [M, H, W]

    '''
    filtered_corners: [K, 3] Filtered corner coordinates
    mask: [N, 8] Boolean mask indicating preserved corner positions
    '''
    after_filter_boxes, mask = filter_3d_corners(corners_array,
                                K,
                                poses,
                                depth_maps,
                                gt_ply,
                                near=0.1,
                                far=100.0,
                                dist_threshold=0.1)
    print("after_filter", after_filter_boxes.shape)
    np.save(os.path.join(data_root,cur_seq,'after_filter_boxes.npy'), after_filter_boxes)