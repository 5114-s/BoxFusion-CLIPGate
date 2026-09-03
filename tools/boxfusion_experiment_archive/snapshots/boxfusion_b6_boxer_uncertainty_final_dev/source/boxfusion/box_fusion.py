import torch
import numpy as np

import cv2

import os
import json
import time
import hashlib

from boxfusion.boxer_uncertainty import (
    fixed_topk_uncertainty_reweighting,
    resolve_boxer_uncertainty_config,
    resolve_final_boxer_uncertainty_config,
    uncertainty_adjusted_selection,
)
from boxfusion.reliable_views import (
    resolve_reliable_view_config,
    select_top_k_reliable_views,
    stable_unique,
    valid_reliable_view_mask,
    weighted_box_initialization,
)

try:
  import pycuda.driver as cuda
  import pycuda.autoprimaryctx
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
  GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  GPU_MODE = 0

class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()
    
class BoxFusion(object):
    def __init__(self, cfg) -> None:
        super(BoxFusion, self).__init__()
        self.cfg = cfg
        self.PST_path = cfg["box_fusion"]["pst_path"]
        self.PST = np.ascontiguousarray(cv2.imread(self.PST_path, -1)) #[3072,6]
        
        self.basedir = cfg['data']['datadir']

        if 'scannet' in self.basedir.lower() or cfg["dataset"] == 'online':
            self.K = np.array([[cfg['cam']['fx'], 0.0, cfg['cam']['cx'],0.0],
                            [0.0, cfg['cam']['fy'], cfg['cam']['cy'],0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]])
            self.H=cfg["cam"]["H"] #l
            self.W=cfg["cam"]["W"] #s

        else: # CA1M
            depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_depth.txt')).reshape(3,3)
            self.K = np.array([[depth_intric[0,0], 0.0, depth_intric[0,2],0.0],
                            [0.0, depth_intric[1,1], depth_intric[1,2],0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]])
            if self.K[0,2] < self.K[1,2]:
                self.H=cfg["cam"]["W"] #l
                self.W=cfg["cam"]["H"] #s
            else:
                self.H=cfg["cam"]["H"]
                self.W=cfg["cam"]["W"]
        self.update_K_flag=False

        self.fusion_iters = cfg["box_fusion"]["iters"]
        self.pst_size = cfg["box_fusion"]["pst_size"]
        self.center_init_size = cfg["box_fusion"]["random_opt"]["center_init_size"]
        self.center_scaling_coefficient = cfg["box_fusion"]["random_opt"]["center_scaling_coefficient"]
        self.shape_init_size = cfg["box_fusion"]["random_opt"]["shape_init_size"]
        self.shape_scaling_coefficient = cfg["box_fusion"]["random_opt"]["shape_scaling_coefficient"]
        self.reliable_view_cfg = resolve_reliable_view_config(
            cfg["box_fusion"]
        )
        self.reliable_view_stats = {
            "fusion_updates": 0,
            "views_available": 0,
            "views_selected": 0,
            "raw_weights": [],
            "selected_weights": [],
            "confidence": [],
            "area_quality": [],
            "projection_iou": [],
            "geometry_consistency": [],
            "invalid_views": 0,
        }
        self.boxer_uncertainty_cfg = resolve_boxer_uncertainty_config(
            self.reliable_view_cfg
        )
        self.final_boxer_uncertainty_cfg = (
            resolve_final_boxer_uncertainty_config(
                self.reliable_view_cfg
            )
        )
        if (
            self.boxer_uncertainty_cfg["mode"] != "disabled"
            and not self.reliable_view_cfg["enabled"]
        ):
            raise ValueError(
                "Boxer uncertainty fusion requires reliable_views.enabled"
            )
        if (
            self.final_boxer_uncertainty_cfg["mode"] != "disabled"
            and not self.reliable_view_cfg["enabled"]
        ):
            raise ValueError(
                "Final Boxer uncertainty requires reliable_views.enabled"
            )
        if (
            self.boxer_uncertainty_cfg["mode"] != "disabled"
            and self.final_boxer_uncertainty_cfg["mode"] != "disabled"
        ):
            raise ValueError(
                "Online and final-only Boxer uncertainty modes are "
                "mutually exclusive"
            )
        self.boxer_uncertainty_stats = {
            "fusion_groups": 0,
            "candidate_views": 0,
            "boxer_views": 0,
            "cutr_fallback_views": 0,
            "invalid_boxer_confidence": 0,
            "candidate_weight_changed_groups": 0,
            "weight_changed_groups": 0,
            "selection_changed_groups": 0,
            "ranking_changed_groups": 0,
            "active_groups": 0,
            "optimization_updated_groups": 0,
            "active_updated_groups": 0,
            "boxer_confidence": [],
            "uncertainty_factors": [],
            "runtime_ms": [],
        }
        normalized_basedir = os.path.normpath(self.basedir)
        self._uncertainty_scene_id = os.path.basename(
            os.path.dirname(normalized_basedir)
        )
        self._uncertainty_records = []
        self._final_uncertainty_recipes = []
        self._final_uncertainty_records = []
        self.final_boxer_uncertainty_stats = {
            "recipes": 0,
            "output_rows": 0,
            "eligible_rows": 0,
            "matched_rows": 0,
            "weight_changed_rows": 0,
            "optimized_rows": 0,
            "applied_rows": 0,
            "selection_changed_rows": 0,
            "ranking_changed_rows": 0,
            "scene_fallback": 0,
            "runtime_ms": 0.0,
            "rejects": {},
        }
        self._final_uncertainty_contract = None



        self.cuda_src_mod = SourceModule("""
            #include <curand_kernel.h>
            #include <algorithm>
            extern "C" {       

            struct Point {
                float x, y;
                __device__ Point(float x=0, float y=0) : x(x), y(y) {}
            };
                                
                                
            __device__ float cross(const Point& o, const Point& a, const Point& b) {
                return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
            }
                                
            __device__ float array_max(float* data, int n) {
                float max_val = data[0];
                for (int i = 1; i < n; i++) {
                    max_val = max(max_val, data[i]);
                }
                return max_val;
            }
                                
            __device__ float array_min(float* data, int n) {
                float min_val = data[0];
                for (int i = 1; i < n; i++) {
                    min_val = min(min_val, data[i]);
                }
                return min_val;
            }
                                
            
            __device__ void convex_hull(Point* in_points, int in_size, 
                            Point* out_points, int& out_size) {
                if (in_size == 0) {
                    out_size = 0;
                    return;
                }
                
                
                for(int i=0; i<in_size-1; ++i){
                    for(int j=i+1; j<in_size; ++j){
                        if(in_points[i].x > in_points[j].x || 
                        (in_points[i].x == in_points[j].x && in_points[i].y > in_points[j].y)){
                            Point tmp = in_points[i];
                            in_points[i] = in_points[j];
                            in_points[j] = tmp;
                        }
                    }
                }

                
                Point* lower = new Point[in_size];
                int lower_size = 0;
                for(int i=0; i<in_size; ++i){
                    while(lower_size >= 2 && 
                        cross(lower[lower_size-2], lower[lower_size-1], in_points[i]) <= 0){
                        lower_size--;
                    }
                    lower[lower_size++] = in_points[i];
                }

                
                Point* upper = new Point[in_size];
                int upper_size = 0;
                for(int i=in_size-1; i>=0; --i){
                    while(upper_size >= 2 && 
                        cross(upper[upper_size-2], upper[upper_size-1], in_points[i]) <= 0){
                        upper_size--;
                    }
                    upper[upper_size++] = in_points[i];
                }

                
                lower_size--;
                upper_size--;
                out_size = lower_size + upper_size;
                for(int i=0; i<lower_size; ++i) out_points[i] = lower[i];
                for(int i=0; i<upper_size; ++i) out_points[lower_size+i] = upper[i];
                
                delete[] lower;
                delete[] upper;
            }
                                
            
            __device__ float polygon_area(Point* poly, int n) {
                float area = 0.0;
                for(int i=0; i<n; ++i){
                    Point& p1 = poly[i];
                    Point& p2 = poly[(i+1)%n];
                    area += p1.x * p2.y - p2.x * p1.y;
                }
                return fabs(area) / 2.0;
            }
                                
            
            __device__ Point* line_intersection(const Point& a1, const Point& a2, 
                                const Point& b1, const Point& b2) {
                double dx1 = a2.x - a1.x;
                double dy1 = a2.y - a1.y;
                double dx2 = b2.x - b1.x;
                double dy2 = b2.y - b1.y;
                
                double denominator = dx1 * dy2 - dy1 * dx2;
                if (std::abs(denominator) < 1e-8) return nullptr;
                
                double t = (dx2*(a1.y - b1.y) + dy2*(b1.x - a1.x)) / denominator;
                double s = (dx1*(a1.y - b1.y) + dy1*(b1.x - a1.x)) / denominator;
                
                if (t >= -1e-8 && t <= 1.00000001 && 
                    s >= -1e-8 && s <= 1.00000001) {
                    return new Point(a1.x + t*dx1, a1.y + t*dy1);
                }
                return nullptr;
            }
                                
            
            __device__ bool point_in_polygon(const Point& p, Point* poly, int poly_size) {
                const float x = p.x;
                const float y = p.y;
                bool inside = false;

                for(int i = 0; i < poly_size; ++i) {
                    const Point& p1 = poly[i];
                    const Point& p2 = poly[(i+1) % poly_size];

                    
                    if( (p1.y > y) != (p2.y > y) ) {
                        
                        const float x_inters = ( (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) ) + p1.x;
                        if(x < x_inters) {
                            inside = !inside;
                        }
                    }
                }
                return inside;
            }
                                
            
            __device__ void polygon_intersection(
                Point* poly1, int poly1_size,
                Point* poly2, int poly2_size,
                Point* candidates, int& cand_size) 
            {
                cand_size = 0;
                
                
                for(int i=0; i<poly1_size; ++i){
                    if(point_in_polygon(poly1[i], poly2, poly2_size)){
                        candidates[(cand_size)++] = poly1[i];
                    }
                }
                for(int i=0; i<poly2_size; ++i){
                    if(point_in_polygon(poly2[i], poly1, poly1_size)){
                        candidates[(cand_size)++] = poly2[i];
                    }
                }
                
                
                for(int i=0; i<poly1_size; ++i){
                    Point a1 = poly1[i];
                    Point a2 = poly1[(i+1)%poly1_size];
                    
                    for(int j=0; j<poly2_size; ++j){
                        Point b1 = poly2[j];
                        Point b2 = poly2[(j+1)%poly2_size];
                        
                        Point* pt = line_intersection(a1, a2, b1, b2);
                        if(pt){
                            candidates[(cand_size)++] = *pt;
                            delete pt;
                        }
                    }
                }
                
                
                if(cand_size > 0){
                    float cx=0, cy=0;
                    for(int i=0; i<cand_size; ++i){
                        cx += candidates[i].x;
                        cy += candidates[i].y;
                    }
                    cx /= cand_size;
                    cy /= cand_size;
                    
                    
                    for(int i=0; i<cand_size-1; ++i){
                        for(int j=0; j<cand_size-i-1; ++j){
                            float angle1 = atan2(candidates[j].y-cy, candidates[j].x-cx);
                            float angle2 = atan2(candidates[j+1].y-cy, candidates[j+1].x-cx);
                            if(angle1 > angle2){
                                Point tmp = candidates[j];
                                candidates[j] = candidates[j+1];
                                candidates[j+1] = tmp;
                            }
                        }
                    }
                }
            }


            __global__ void compute_iou_value(float * box_3d,
                                    float * t_c,
                                    float * scores,    
                                    float * transform_candidate,
                                    float * box_rot,
                                    float * cam_poses,
                                    float * K,
                                    float * search_size,
                                    float * search_value,
                                    float * search_count,
                                    float * other_params
                                ){
           
            int node=blockDim.x*blockIdx.x+threadIdx.x;
            
            
            float img_h = other_params[0];
            float img_w = other_params[1];
            float node_size = other_params[2];
            int num_boxes = (int) other_params[3];
            bool use_view_weights = other_params[4] > 0.5f;
            
            if (node>=node_size){
                return;
            }

            float x3d = box_3d[0];
            float y3d = box_3d[1];
            float z3d = box_3d[2];
            float w3d = box_3d[5];
            float h3d = box_3d[4];
            float l3d = box_3d[3];
                            
            x3d = x3d + transform_candidate[node*6+0] * search_size[0];
            y3d = y3d + transform_candidate[node*6+1] * search_size[1];
            z3d = z3d + transform_candidate[node*6+2] * search_size[2];
            w3d = w3d + transform_candidate[node*6+5] * search_size[5];
            h3d = h3d + transform_candidate[node*6+4] * search_size[4];
            l3d = l3d + transform_candidate[node*6+3] * search_size[3];
            
            float xyz[3] = {x3d,y3d,z3d};

            w3d = max(w3d, 0.01f); 
            h3d = max(h3d, 0.01f); 
            l3d = max(l3d, 0.01f);                           


            float verts[8][3] = {
            {-l3d / 2, -h3d / 2, -w3d / 2},
            {l3d / 2, -h3d / 2, -w3d / 2},                
            {l3d / 2, h3d / 2, -w3d / 2},
            {-l3d / 2, h3d / 2, -w3d / 2},
            {-l3d / 2, -h3d / 2, w3d / 2},
            {l3d / 2, -h3d / 2, w3d / 2},
            {l3d / 2, h3d / 2, w3d / 2},
            {-l3d / 2, h3d / 2, w3d / 2},
            };
                            
            
            float corners[8][3] = {0}; 

            for (int i =0; i<8; ++i){          
                for (int j=0; j<3; ++j){       
                    for (int k=0; k<3; ++k){  
                        corners[i][j] += box_rot[j*3+k] * verts[i][k];        
                    } 
                    corners[i][j] += xyz[j];
                }  
            } 

            
            
            //project pts in world cordinate into 2D planes and get [u,v] -> [N,8,2]
                            
            int i=(blockDim.y*blockIdx.y+threadIdx.y);
                            
            if (i>=num_boxes){
                return;
            }
                                         
            float score_box = scores[i];
            float view_weight = use_view_weights
                ? max(score_box, 0.000001f)
                : 1.0f;
                           
            float uv[8][2] = {0};
                        
            for (int j=0; j<8; ++j){ 
                float vertex_x = corners[j][0]-cam_poses[i*16+3];
                float vertex_y = corners[j][1]-cam_poses[i*16+7];
                float vertex_z = corners[j][2]-cam_poses[i*16+11];
                
                float cam_x = cam_poses[i*16+0]*vertex_x+cam_poses[i*16+4]*vertex_y+cam_poses[i*16+8]*vertex_z ;
                float cam_y = cam_poses[i*16+1]*vertex_x+cam_poses[i*16+5]*vertex_y+cam_poses[i*16+9]*vertex_z ;
                float cam_z = cam_poses[i*16+2]*vertex_x+cam_poses[i*16+6]*vertex_y+cam_poses[i*16+10]*vertex_z ;

                float pixel_x = ((cam_x*K[0])/cam_z+K[2]);
                float pixel_y = ((cam_y*K[5])/cam_z+K[6]);
                
                uv[j][0] = (pixel_x > img_w) ? img_w : (pixel_x < 0) ? 0 : pixel_x;
                uv[j][1] = (pixel_y > img_h) ? img_h : (pixel_y < 0) ? 0 : pixel_y;
            }
                            

            Point corners0[8] = {Point(uv[0][0], uv[0][1]),Point(uv[1][0], uv[1][1]),Point(uv[2][0], uv[2][1]),Point(uv[3][0], uv[3][1]),Point(uv[4][0], uv[4][1]),Point(uv[5][0], uv[5][1]),Point(uv[6][0], uv[6][1]),Point(uv[7][0], uv[7][1])};  
            

            Point t_corners0[8] = {Point(t_c[i*16+0], t_c[i*16+1]),Point(t_c[i*16+2], t_c[i*16+3]),Point(t_c[i*16+4], t_c[i*16+5]),Point(t_c[i*16+6], t_c[i*16+7]),Point(t_c[i*16+8], t_c[i*16+9]),Point(t_c[i*16+10], t_c[i*16+11]),Point(t_c[i*16+12], t_c[i*16+13]),Point(t_c[i*16+14], t_c[i*16+15])};

            Point convex_0[8]; 
            int out_size_0 = 8;
            Point convex_t[8]; 
            int out_size_t = 8;

            convex_hull(corners0, 8, convex_0, out_size_0);
            convex_hull(t_corners0, 8, convex_t, out_size_t);


            Point corners_i[36]; 
            int corners_i_size = 8;
            polygon_intersection(convex_0,out_size_0,convex_t,out_size_t,corners_i,corners_i_size);
            
            Point convex_inter[8]; 
            int out_size_inter = 8;
            convex_hull(corners_i, corners_i_size, convex_inter, out_size_inter);
                            
                            
            float inter_area = polygon_area(convex_inter, out_size_inter);
            float area0 = polygon_area(convex_0, out_size_0);
            float area_t = polygon_area(convex_t, out_size_t);
            
                
            float union_area = area0 + area_t - inter_area;
            float iou = 0;         
            if (union_area>0){
                iou =  inter_area / (union_area+0.00001);  
                
                iou = iou; //* max(score_box,1.0);
            }
            
            if (use_view_weights){
                atomicAdd_system(
                    search_value+node,
                    view_weight*abs(1-iou)
                );
                atomicAdd_system(search_count+node,view_weight);
            }
            else{
                // Keep the released objective byte-for-byte at the arithmetic
                // operation level when reliable-view fusion is disabled.
                atomicAdd_system(search_value+node,abs(1-iou));
                atomicAdd_system(search_count+node,1);
            }
                        
            return;

        }
}
         """, no_extern_c=True)

        self.cuda_compute_iou_value = self.cuda_src_mod.get_function("compute_iou_value") 



    def evaluate_iou(self, 
                    box_3d,
                    corners_2d,
                    box_rot,
                    scores_box,
                    camera_poses,
                    search_size,
                    num_of_boxes,
                    verbose=False,
                    use_view_weights=False):

        search_value=np.zeros((self.PST.shape[0])).astype(np.float32)
        search_count=np.zeros((self.PST.shape[0])).astype(np.float32)
        if verbose:
            print("box_3d",box_3d)
            print("corners_2d",corners_2d)
            print("self.PST",self.PST)
            print("box_rot",box_rot)
            print("camera_poses",camera_poses)
            print("search_size",search_size)
        self.cuda_compute_iou_value(
                        cuda.In(box_3d.reshape(-1).astype(np.float32)),
                        cuda.In(corners_2d.reshape(-1).astype(np.float32)),
                        cuda.In(scores_box.reshape(-1).astype(np.float32)),
                        cuda.In(self.PST.reshape(-1).astype(np.float32)),
                        cuda.In(box_rot.reshape(-1).astype(np.float32)),
                        cuda.In(camera_poses.reshape(-1).astype(np.float32)),
                        cuda.In(self.K.reshape(-1).astype(np.float32)),
                        cuda.In(search_size),
                        cuda.InOut(search_value),
                        cuda.InOut(search_count),
                        cuda.In(np.asarray([
                                        self.H,
                                        self.W,
                                        self.pst_size,
                                        num_of_boxes,
                                        float(use_view_weights)
                                        ], np.float32)),
           
                        block=(32,1,1),  
                        grid=( int(self.pst_size/(32)),num_of_boxes,1)  # 3,1      
                        )
        
        fitness = search_value/(search_count+1e-6)

        if verbose:

            print("search value",search_value, search_value.shape, np.sum(search_value), 'last best iou:',1-fitness[0])


        return fitness

    def record_reliable_view_selection(self, selection):
        self.reliable_view_stats["fusion_updates"] += 1
        self.reliable_view_stats["views_available"] += int(
            selection["weights"].shape[0]
        )
        self.reliable_view_stats["views_selected"] += int(
            selection["selected_indices"].shape[0]
        )
        self.reliable_view_stats["raw_weights"].extend(
            selection["weights"].tolist()
        )
        self.reliable_view_stats["selected_weights"].extend(
            selection["selected_weights"].tolist()
        )
        for key in (
            "confidence",
            "area_quality",
            "projection_iou",
            "geometry_consistency",
        ):
            self.reliable_view_stats[key].extend(
                selection[key].tolist()
            )

    def reliable_view_summary(self):
        stats = self.reliable_view_stats

        def quantiles(values):
            values = np.asarray(values, dtype=np.float32)
            if values.size == 0:
                return "nan/nan/nan"
            q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
            return f"{q10:.4f}/{q50:.4f}/{q90:.4f}"

        return (
            "Reliable-view fusion summary | "
            f"updates={stats['fusion_updates']}, "
            f"available={stats['views_available']}, "
            f"selected={stats['views_selected']}, "
            "weight_q10/q50/q90="
            f"{quantiles(stats['raw_weights'])}, "
            "selected_q10/q50/q90="
            f"{quantiles(stats['selected_weights'])}, "
            "confidence_q10/q50/q90="
            f"{quantiles(stats['confidence'])}, "
            "area_q10/q50/q90="
            f"{quantiles(stats['area_quality'])}, "
            "projection_iou_q10/q50/q90="
            f"{quantiles(stats['projection_iou'])}, "
            "geometry_q10/q50/q90="
            f"{quantiles(stats['geometry_consistency'])}, "
            f"invalid={stats['invalid_views']}"
        )

    def record_boxer_uncertainty_selection(
        self,
        fusion_index,
        candidate_indices,
        base_selection,
        adjusted_selection,
        runtime_ms,
    ):
        """Record observer/active decisions without touching fusion state."""

        stats = self.boxer_uncertainty_stats
        applied = np.asarray(
            adjusted_selection["boxer_geometry_applied"], dtype=bool
        )
        valid_confidence = np.asarray(
            adjusted_selection["boxer_confidence_valid"], dtype=bool
        )
        factors = np.asarray(
            adjusted_selection["uncertainty_factors"], dtype=np.float32
        )
        confidence = np.asarray(
            adjusted_selection["boxer_confidence"], dtype=np.float32
        )
        weighted = np.asarray(
            adjusted_selection["uncertainty_weighted_rows"], dtype=bool
        )
        selection_changed = bool(
            np.asarray(adjusted_selection["selection_changed"]).item()
        )
        ranking_changed = bool(
            np.asarray(adjusted_selection["ranking_changed"]).item()
        )
        candidate_weights_changed = bool(
            np.asarray(
                adjusted_selection["candidate_weights_changed"]
            ).item()
        )
        weights_changed = bool(
            np.asarray(
                adjusted_selection["effective_weights_changed"]
            ).item()
        )

        stats["fusion_groups"] += 1
        stats["candidate_views"] += int(applied.shape[0])
        stats["boxer_views"] += int(np.count_nonzero(applied))
        stats["cutr_fallback_views"] += int(np.count_nonzero(~applied))
        stats["invalid_boxer_confidence"] += int(
            np.count_nonzero(applied & ~valid_confidence)
        )
        stats["candidate_weight_changed_groups"] += int(
            candidate_weights_changed
        )
        stats["weight_changed_groups"] += int(weights_changed)
        stats["selection_changed_groups"] += int(selection_changed)
        stats["ranking_changed_groups"] += int(ranking_changed)
        stats["active_groups"] += int(
            self.boxer_uncertainty_cfg["mode"] == "active"
            and weights_changed
        )
        stats["boxer_confidence"].extend(confidence[weighted].tolist())
        stats["uncertainty_factors"].extend(factors[weighted].tolist())
        stats["runtime_ms"].append(float(runtime_ms))

        candidate_indices = np.asarray(
            candidate_indices, dtype=np.int64
        ).reshape(-1)
        base_selected = np.asarray(
            base_selection["selected_indices"], dtype=np.int64
        )
        adjusted_selected = np.asarray(
            adjusted_selection["selected_indices"], dtype=np.int64
        )
        # Invalid values are deliberately fail-neutral for fusion.  Encode
        # them as JSON null rather than NaN/Inf so allow_nan=False remains a
        # strict serialization guard; boxer_confidence_valid retains the
        # exact row-level validity information.
        confidence_json = [
            float(value) if bool(is_valid) else None
            for value, is_valid in zip(confidence, valid_confidence)
        ]
        record = {
                "fusion_update": int(stats["fusion_groups"] - 1),
                "global_box_index": int(fusion_index),
                "global_stable_id": int(candidate_indices.min()),
                "mode": self.boxer_uncertainty_cfg["mode"],
                "candidate_indices": candidate_indices.tolist(),
                "base_selected_indices": candidate_indices[
                    base_selected
                ].tolist(),
                "uncertainty_selected_indices": candidate_indices[
                    adjusted_selected
                ].tolist(),
                "base_weights": np.asarray(
                    adjusted_selection["base_weights"], dtype=np.float32
                ).tolist(),
                "uncertainty_weights": np.asarray(
                    adjusted_selection["uncertainty_weights"],
                    dtype=np.float32,
                ).tolist(),
                "uncertainty_factors": factors.tolist(),
                "base_effective_weights": np.asarray(
                    adjusted_selection["base_effective_weights"],
                    dtype=np.float32,
                ).tolist(),
                "uncertainty_effective_weights": np.asarray(
                    adjusted_selection["uncertainty_effective_weights"],
                    dtype=np.float32,
                ).tolist(),
                "boxer_confidence": confidence_json,
                "boxer_geometry_applied": applied.tolist(),
                "boxer_confidence_valid": valid_confidence.tolist(),
                "selection_changed": selection_changed,
                "ranking_changed": ranking_changed,
                "candidate_weights_changed": candidate_weights_changed,
                "weights_changed": weights_changed,
                "applied_to_fusion": bool(
                    self.boxer_uncertainty_cfg["mode"] == "active"
                ),
                "optimization_updated": False,
                "runtime_ms": float(runtime_ms),
            }
        self._uncertainty_records.append(record)
        return record

    def _write_boxer_uncertainty_diagnostics(self):
        root = self.boxer_uncertainty_cfg["diagnostics_dir"]
        if not root or self.boxer_uncertainty_cfg["mode"] == "disabled":
            return None
        os.makedirs(root, exist_ok=True)
        output_path = os.path.join(
            root,
            f"{self._uncertainty_scene_id}_boxer_uncertainty.json",
        )
        payload = {
            "schema": "boxfusion.boxer_uncertainty_fusion.scene.v1",
            "scene_id": self._uncertainty_scene_id,
            "config": dict(self.boxer_uncertainty_cfg),
            "summary": {
                key: value
                for key, value in self.boxer_uncertainty_stats.items()
                if not isinstance(value, list)
            },
            "records": self._uncertainty_records,
        }
        temporary_path = f"{output_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, output_path)
        return output_path

    def boxer_uncertainty_summary(self):
        """Persist diagnostics and return a compact scene summary."""

        stats = self.boxer_uncertainty_stats

        def quantiles(values):
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0:
                return "nan/nan/nan"
            q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
            return f"{q10:.4f}/{q50:.4f}/{q90:.4f}"

        diagnostic_path = self._write_boxer_uncertainty_diagnostics()
        runtime = np.asarray(stats["runtime_ms"], dtype=np.float64)
        runtime_total = float(runtime.sum()) if runtime.size else 0.0
        return (
            "Boxer uncertainty fusion summary | "
            f"mode={self.boxer_uncertainty_cfg['mode']}, "
            f"groups={stats['fusion_groups']}, "
            f"views(boxer/fallback)={stats['boxer_views']}/"
            f"{stats['cutr_fallback_views']}, "
            f"weight_changed={stats['weight_changed_groups']}, "
            "candidate_weight_changed="
            f"{stats['candidate_weight_changed_groups']}, "
            f"selection_changed={stats['selection_changed_groups']}, "
            f"ranking_changed={stats['ranking_changed_groups']}, "
            f"active_groups={stats['active_groups']}, "
            f"active_updated={stats['active_updated_groups']}, "
            f"invalid_confidence={stats['invalid_boxer_confidence']}, "
            "confidence_q10/q50/q90="
            f"{quantiles(stats['boxer_confidence'])}, "
            "factor_q10/q50/q90="
            f"{quantiles(stats['uncertainty_factors'])}, "
            f"runtime_ms={runtime_total:.3f}, "
            f"diagnostic={diagnostic_path or 'disabled'}"
        )

    @staticmethod
    def _array_digest(values):
        array = np.asarray(values)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _obb_corners_numpy(xyzlwh, rotation):
        box = np.asarray(xyzlwh, dtype=np.float32).reshape(6)
        matrix = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        length, height, width = box[3:6]
        vertices = np.asarray(
            [
                [-length / 2, -height / 2, -width / 2],
                [length / 2, -height / 2, -width / 2],
                [length / 2, height / 2, -width / 2],
                [-length / 2, height / 2, -width / 2],
                [-length / 2, -height / 2, width / 2],
                [length / 2, -height / 2, width / 2],
                [length / 2, height / 2, width / 2],
                [-length / 2, height / 2, width / 2],
            ],
            dtype=np.float32,
        )
        return (vertices @ matrix.T + box[:3]).astype(np.float32)

    def record_final_boxer_uncertainty_recipe(
        self,
        *,
        global_box_index,
        source_group,
        candidate_indices,
        base_selection,
        projected_corners,
        camera_poses,
        boxer_confidence,
        boxer_geometry_applied,
    ):
        """Freeze the exact G0 Top-K recipe without mutating online fusion."""

        if self.final_boxer_uncertainty_cfg["mode"] == "disabled":
            return
        candidate_indices = np.asarray(
            candidate_indices, dtype=np.int64
        ).reshape(-1)
        selected_local = np.asarray(
            base_selection["selected_indices"], dtype=np.int64
        ).reshape(-1)
        if (
            candidate_indices.size == 0
            or selected_local.size == 0
            or np.any(selected_local < 0)
            or np.any(selected_local >= candidate_indices.size)
        ):
            raise ValueError("invalid G0 reliable-view recipe")

        projected_corners = np.asarray(
            projected_corners, dtype=np.float32
        )
        camera_poses = np.asarray(camera_poses, dtype=np.float32)
        boxer_confidence = np.asarray(
            boxer_confidence, dtype=np.float32
        ).reshape(-1)
        boxer_geometry_applied = np.asarray(
            boxer_geometry_applied, dtype=bool
        ).reshape(-1)
        count = candidate_indices.size
        if (
            projected_corners.shape != (count, 8, 2)
            or camera_poses.shape != (count, 4, 4)
            or boxer_confidence.shape[0] != count
            or boxer_geometry_applied.shape[0] != count
        ):
            raise ValueError("G0 recipe arrays are not row aligned")

        frozen_selection = {
            key: np.asarray(value).copy()
            if isinstance(value, np.ndarray)
            else value
            for key, value in base_selection.items()
        }
        recipe = {
            "sequence": len(self._final_uncertainty_recipes),
            "global_box_index_at_fusion": int(global_box_index),
            "source_group": tuple(int(value) for value in source_group),
            "candidate_indices": candidate_indices.copy(),
            "selected_source_indices": candidate_indices[
                selected_local
            ].copy(),
            "base_selection": frozen_selection,
            "projected_corners": projected_corners.copy(),
            "camera_poses": camera_poses.copy(),
            "boxer_confidence": boxer_confidence.copy(),
            "boxer_geometry_applied": boxer_geometry_applied.copy(),
        }
        self._final_uncertainty_recipes.append(recipe)
        self.final_boxer_uncertainty_stats["recipes"] = len(
            self._final_uncertainty_recipes
        )

    def _find_final_uncertainty_recipe(self, source_group):
        group = frozenset(int(value) for value in source_group)
        exact = [
            recipe
            for recipe in self._final_uncertainty_recipes
            if tuple(recipe["source_group"]) == tuple(source_group)
        ]
        if exact:
            return max(exact, key=lambda item: int(item["sequence"]))
        compatible = [
            recipe
            for recipe in self._final_uncertainty_recipes
            if frozenset(recipe["source_group"]).issubset(group)
            and frozenset(recipe["selected_source_indices"]).issubset(group)
        ]
        if not compatible:
            return None
        return max(
            compatible,
            key=lambda item: (
                len(item["source_group"]),
                int(item["sequence"]),
            ),
        )

    def _local_search_size_update(self, iou, mean_transform):
        min_scale = 1e-3
        transform = np.asarray(mean_transform, dtype=np.float32).reshape(6)
        magnitudes = np.abs(transform) + min_scale
        norm = float(np.linalg.norm(magnitudes))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("invalid final-only search-size norm")
        normalized = magnitudes / norm
        updated = np.empty(6, dtype=np.float32)
        updated[:3] = (
            self.center_scaling_coefficient
            * float(iou)
            * normalized[:3]
            + min_scale
        )
        updated[3:] = (
            self.shape_scaling_coefficient
            * float(iou)
            * normalized[3:]
            + min_scale
        )
        return updated

    def _optimize_final_uncertainty_candidate(
        self,
        *,
        initial_xyzlwh,
        fixed_rotation,
        projected_corners,
        camera_poses,
        view_weights,
        beta=0.9,
    ):
        """Run the released objective with entirely local optimization state."""

        global_xyzlwh = np.asarray(
            initial_xyzlwh, dtype=np.float32
        ).reshape(6).copy()
        fixed_rotation = np.asarray(
            fixed_rotation, dtype=np.float32
        ).reshape(3, 3).copy()
        projected_corners = np.asarray(
            projected_corners, dtype=np.float32
        )
        camera_poses = np.asarray(camera_poses, dtype=np.float32)
        view_weights = np.asarray(
            view_weights, dtype=np.float32
        ).reshape(-1)
        count = view_weights.shape[0]
        if (
            projected_corners.shape != (count, 8, 2)
            or camera_poses.shape != (count, 4, 4)
            or count == 0
        ):
            raise ValueError("final-only optimization arrays are misaligned")
        for values in (
            global_xyzlwh,
            fixed_rotation,
            projected_corners,
            camera_poses,
            view_weights,
        ):
            if not np.isfinite(values).all():
                raise ValueError("final-only optimization inputs must be finite")
        if np.any(global_xyzlwh[3:] <= 0.0) or np.any(view_weights <= 0.0):
            raise ValueError("final-only sizes and weights must be positive")

        search_size = np.empty(6, dtype=np.float32)
        search_size[:3] = self.center_init_size
        search_size[3:] = self.shape_init_size
        previous_search_size = np.zeros(6, dtype=np.float32)
        need_update = False
        previous_success = False
        fail_count = 0
        iterations = 0

        for _ in range(self.fusion_iters):
            iterations += 1
            search_value = self.evaluate_iou(
                global_xyzlwh,
                projected_corners,
                fixed_rotation,
                view_weights,
                camera_poses,
                search_size,
                count,
                verbose=False,
                use_view_weights=True,
            )
            if not np.isfinite(search_value).all():
                raise ValueError("non-finite final-only objective")
            success, minimum_iou, mean_transform = self.cal_transform(
                search_value, search_size
            )
            updated_search_size = self._local_search_size_update(
                minimum_iou, mean_transform
            )
            if previous_success and success:
                updated_search_size = (
                    beta * updated_search_size
                    + (1.0 - beta) * previous_search_size
                ).astype(np.float32)
            search_size = updated_search_size

            if success:
                need_update = True
                previous_success = True
                fail_count = 0
                global_xyzlwh += mean_transform
                previous_search_size = search_size.copy()
            else:
                fail_count += 1
                previous_success = False
            if fail_count >= 3:
                break

        global_xyzlwh[3:] = np.maximum(global_xyzlwh[3:], 0.01)
        if not np.isfinite(global_xyzlwh).all():
            raise ValueError("final-only optimizer produced non-finite geometry")
        return global_xyzlwh, fixed_rotation, bool(need_update), iterations

    def apply_final_boxer_uncertainty(
        self,
        *,
        baseline_corners,
        scores,
        source_indices,
        stable_ids,
        global_xyzlwh,
        global_rotations,
        global_stable_ids,
        frozen_fusion_groups,
        minimum_extent,
    ):
        """Return post-B6 geometry while preserving all protected fields."""

        mode = self.final_boxer_uncertainty_cfg["mode"]
        baseline = np.asarray(baseline_corners, dtype=np.float32)
        protected_scores = np.asarray(scores)
        sources = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        output_ids = np.asarray(stable_ids, dtype=np.int64).reshape(-1)
        global_boxes = np.asarray(global_xyzlwh, dtype=np.float32)
        global_r = np.asarray(global_rotations, dtype=np.float32)
        global_ids = np.asarray(global_stable_ids, dtype=np.int64).reshape(-1)
        groups = tuple(tuple(int(v) for v in group) for group in frozen_fusion_groups)

        if baseline.ndim != 3 or baseline.shape[1:] != (8, 3):
            raise ValueError("baseline_corners must have shape [N, 8, 3]")
        row_count = baseline.shape[0]
        if not (
            protected_scores.shape[0]
            == sources.shape[0]
            == output_ids.shape[0]
            == row_count
        ):
            raise ValueError("final-only output arrays must be row aligned")
        if global_boxes.shape != (len(groups), 6):
            raise ValueError("global_xyzlwh must align with fusion groups")
        if global_r.shape != (len(groups), 3, 3):
            raise ValueError("global_rotations must align with fusion groups")
        if global_ids.shape[0] != len(groups):
            raise ValueError("global_stable_ids must align with fusion groups")
        if not np.isfinite(baseline).all() or not np.isfinite(global_boxes).all():
            raise ValueError("final-only geometry inputs must be finite")
        minimum_extent = float(minimum_extent)
        if not np.isfinite(minimum_extent) or minimum_extent < 0.0:
            raise ValueError("minimum_extent must be finite and non-negative")

        output = baseline.copy()
        started = time.perf_counter()
        stats = self.final_boxer_uncertainty_stats
        stats["output_rows"] = row_count
        records = []

        def reject(reason):
            rejects = stats["rejects"]
            rejects[reason] = int(rejects.get(reason, 0)) + 1

        for row in range(row_count):
            source_index = int(sources[row])
            stable_id = int(output_ids[row])
            record = {
                "row": row,
                "source_index": source_index,
                "stable_id": stable_id,
                "mode": mode,
                "applied": False,
            }
            if source_index < 0:
                record["reason"] = "supplemental"
                reject("supplemental")
                records.append(record)
                continue
            stats["eligible_rows"] += 1
            if source_index >= len(groups):
                record["reason"] = "source_index_out_of_range"
                reject(record["reason"])
                records.append(record)
                continue
            if stable_id != int(global_ids[source_index]):
                record["reason"] = "stable_id_mismatch"
                reject(record["reason"])
                records.append(record)
                continue

            initial = global_boxes[source_index]
            rotation = global_r[source_index]
            initial_corners = self._obb_corners_numpy(initial, rotation)
            if not np.allclose(
                initial_corners,
                baseline[row],
                rtol=1e-5,
                atol=1e-5,
            ):
                record["reason"] = "baseline_geometry_mismatch"
                reject(record["reason"])
                records.append(record)
                continue
            recipe = self._find_final_uncertainty_recipe(
                groups[source_index]
            )
            if recipe is None:
                record["reason"] = "recipe_missing"
                reject(record["reason"])
                records.append(record)
                continue
            stats["matched_rows"] += 1

            try:
                adjusted = fixed_topk_uncertainty_reweighting(
                    recipe["base_selection"],
                    recipe["boxer_confidence"],
                    recipe["boxer_geometry_applied"],
                    self.final_boxer_uncertainty_cfg,
                )
                selected_local = np.asarray(
                    adjusted["selected_indices"], dtype=np.int64
                )
                base_selected = np.asarray(
                    adjusted["base_selected_indices"], dtype=np.int64
                )
                selection_changed = not np.array_equal(
                    selected_local, base_selected
                )
                ranking_changed = bool(
                    np.asarray(adjusted["ranking_changed"]).item()
                )
                stats["selection_changed_rows"] += int(selection_changed)
                stats["ranking_changed_rows"] += int(ranking_changed)
                if selection_changed or ranking_changed:
                    raise ValueError("fixed Top-K contract was violated")

                changed = bool(
                    np.asarray(
                        adjusted["effective_weights_changed"]
                    ).item()
                )
                record.update(
                    {
                        "recipe_sequence": int(recipe["sequence"]),
                        "source_group": list(groups[source_index]),
                        "selected_source_indices": np.asarray(
                            recipe["selected_source_indices"],
                            dtype=np.int64,
                        ).tolist(),
                        "base_effective_weights": np.asarray(
                            adjusted["base_effective_weights"],
                            dtype=np.float32,
                        )[selected_local].tolist(),
                        "uncertainty_effective_weights": np.asarray(
                            adjusted["uncertainty_effective_weights"],
                            dtype=np.float32,
                        )[selected_local].tolist(),
                        "weights_changed": changed,
                        "selection_changed": selection_changed,
                        "ranking_changed": ranking_changed,
                    }
                )
                if not changed:
                    record["reason"] = "weights_unchanged"
                    reject(record["reason"])
                    records.append(record)
                    continue
                stats["weight_changed_rows"] += 1

                selected_weights = np.asarray(
                    adjusted["selected_weights"], dtype=np.float32
                )
                selected_weights = selected_weights / max(
                    float(selected_weights.mean()), 1e-6
                )
                selected_corners = np.asarray(
                    recipe["projected_corners"], dtype=np.float32
                )[selected_local]
                selected_poses = np.asarray(
                    recipe["camera_poses"], dtype=np.float32
                )[selected_local]
                candidate, fixed_r, optimized, iterations = (
                    self._optimize_final_uncertainty_candidate(
                        initial_xyzlwh=initial,
                        fixed_rotation=rotation,
                        projected_corners=selected_corners,
                        camera_poses=selected_poses,
                        view_weights=selected_weights,
                    )
                )
                record["iterations"] = int(iterations)
                record["optimized"] = bool(optimized)
                if not optimized:
                    record["reason"] = "no_improving_particle"
                    reject(record["reason"])
                    records.append(record)
                    continue
                stats["optimized_rows"] += 1
                candidate_corners = self._obb_corners_numpy(
                    candidate, fixed_r
                )
                candidate_extents = np.ptp(candidate_corners, axis=0)
                if (
                    not np.isfinite(candidate_corners).all()
                    or not np.isfinite(candidate_extents).all()
                    or np.any(candidate[3:] <= 0.0)
                ):
                    raise ValueError("candidate geometry is invalid")
                if np.any(candidate_extents < minimum_extent):
                    record["reason"] = "candidate_below_minimum_extent"
                    reject(record["reason"])
                    records.append(record)
                    continue

                center_shift = float(
                    np.linalg.norm(candidate[:3] - initial[:3])
                )
                volume_ratio = float(
                    np.prod(candidate[3:]) / np.prod(initial[3:])
                )
                record.update(
                    {
                        "reason": (
                            "active_geometry_replaced"
                            if mode == "active"
                            else "observer_candidate"
                        ),
                        "center_shift_m": center_shift,
                        "volume_ratio": volume_ratio,
                        "baseline_corners": baseline[row].tolist(),
                        "candidate_corners": candidate_corners.tolist(),
                    }
                )
                if mode == "active":
                    output[row] = candidate_corners
                    record["applied"] = True
                    stats["applied_rows"] += 1
                records.append(record)
            except (ValueError, FloatingPointError) as error:
                record["reason"] = "candidate_error"
                record["error"] = str(error)
                reject(record["reason"])
                records.append(record)

        stats["runtime_ms"] = float(
            (time.perf_counter() - started) * 1000.0
        )
        self._final_uncertainty_records = records

        scores_before = self._array_digest(protected_scores)
        source_before = self._array_digest(sources)
        ids_before = self._array_digest(output_ids)
        protected_ok = (
            output.shape == baseline.shape
            and self._array_digest(protected_scores) == scores_before
            and self._array_digest(sources) == source_before
            and self._array_digest(output_ids) == ids_before
            and (
                mode != "observer"
                or np.array_equal(output, baseline)
            )
        )
        if not protected_ok:
            output = baseline.copy()
            stats["scene_fallback"] = 1
            stats["applied_rows"] = 0
            for record in records:
                record["applied"] = False

        self._final_uncertainty_contract = {
            "protected_fields_equal": bool(protected_ok),
            "scene_fallback": bool(not protected_ok),
            "count_before": int(row_count),
            "count_after": int(output.shape[0]),
            "scores_sha256_before": scores_before,
            "scores_sha256_after": self._array_digest(protected_scores),
            "source_indices_sha256_before": source_before,
            "source_indices_sha256_after": self._array_digest(sources),
            "stable_ids_sha256_before": ids_before,
            "stable_ids_sha256_after": self._array_digest(output_ids),
            "baseline_corners_sha256": self._array_digest(baseline),
            "output_corners_sha256": self._array_digest(output),
        }
        return output

    def _write_final_boxer_uncertainty_diagnostics(self):
        cfg = self.final_boxer_uncertainty_cfg
        if cfg["mode"] == "disabled" or not cfg["diagnostics_dir"]:
            return None
        if self._final_uncertainty_contract is None:
            raise RuntimeError(
                "final Boxer uncertainty was not applied before diagnostics"
            )
        root = cfg["diagnostics_dir"]
        os.makedirs(root, exist_ok=True)
        output_path = os.path.join(
            root,
            f"{self._uncertainty_scene_id}_final_boxer_uncertainty.json",
        )
        payload = {
            "schema": "boxfusion.final_boxer_uncertainty.scene.v1",
            "scene_id": self._uncertainty_scene_id,
            "config": dict(cfg),
            "summary": dict(self.final_boxer_uncertainty_stats),
            "contract": dict(self._final_uncertainty_contract),
            "records": self._final_uncertainty_records,
        }
        temporary_path = f"{output_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, output_path)
        return output_path

    def final_boxer_uncertainty_summary(self):
        path = self._write_final_boxer_uncertainty_diagnostics()
        stats = self.final_boxer_uncertainty_stats
        return (
            "Final-only Boxer uncertainty summary | "
            f"mode={self.final_boxer_uncertainty_cfg['mode']}, "
            f"recipes={stats['recipes']}, "
            f"rows={stats['output_rows']}, "
            f"matched={stats['matched_rows']}, "
            f"weight_changed={stats['weight_changed_rows']}, "
            f"optimized={stats['optimized_rows']}, "
            f"applied={stats['applied_rows']}, "
            f"selection/ranking_changed="
            f"{stats['selection_changed_rows']}/"
            f"{stats['ranking_changed_rows']}, "
            f"fallback={stats['scene_fallback']}, "
            f"runtime_ms={stats['runtime_ms']:.3f}, "
            f"diagnostic={path or 'disabled'}"
        )

    def update_intrinsics(self,size,K):
        self.H=size[1]
        self.W=size[0]
        self.K[:3,:3] = K

    def init_searchsize(self):
        self.search_size=np.zeros((6),dtype=np.float32)
        self.previous_search_size =np.zeros((6),dtype=np.float32)
        self.search_size[:3] = self.center_init_size
        self.search_size[3:] = self.shape_init_size


    def cal_transform(self,search_value,search_size):
        # calculate the mean_transform result:
        mean_transform = np.zeros((6),dtype=np.float32) 
        origin_iou = search_value[0]
        # init sum value
        sum_tx = 0.0
        sum_ty = 0.0
        sum_tz = 0.0
        sum_l = 0.0
        sum_w = 0.0
        sum_h = 0.0
        sum_weight = 0.0
        sum_iou = 0.0
        count_search = 0

        for j in range(1,len(search_value)):

            if search_value[j]<origin_iou:
                tx = self.PST[j][0]
                ty = self.PST[j][1]
                tz = self.PST[j][2]
                qx = self.PST[j][3]
                qy = self.PST[j][4]
                qz = self.PST[j][5]
                cur_fit = search_value[j]
                weight = origin_iou - cur_fit

                sum_tx +=tx*weight
                sum_ty +=ty*weight
                sum_tz +=tz*weight
                sum_l +=qx*weight
                sum_w +=qy*weight
                sum_h +=qz*weight
                
                sum_weight +=weight
                sum_iou +=cur_fit*weight
                count_search +=1

                
                if count_search== 200:
                    break 
                
        # If all particles are consistently worse than particle 0, skip this round. If all are worse, keep the best pose from previous frame
        if count_search <= 0:
            success = False
            min_iou = origin_iou #* DIVSHORTMAX
            return False,min_iou,mean_transform
        #
        mean_iou = sum_iou / sum_weight
        mean_transform[0] = (sum_tx / sum_weight)*search_size[0]
        mean_transform[1] = (sum_ty / sum_weight)*search_size[1]
        mean_transform[2] = (sum_tz / sum_weight)*search_size[2]
    

        mean_transform[3] = (sum_l / sum_weight)*search_size[3]
        mean_transform[4] = (sum_w / sum_weight)*search_size[4]
        mean_transform[5] = (sum_h / sum_weight)*search_size[5]

        min_tsdf = mean_iou #* DIVSHORTMAX

        return True,min_tsdf,mean_transform

    def update_PST(self, iou,mean_transform,min_scale=1e-3,center_scale=0.5, shape_scale=0.5): #min_scale=1e-3
        
        s_tx =abs(mean_transform[0])+min_scale
        s_ty =abs(mean_transform[1])+min_scale
        s_tz =abs(mean_transform[2])+min_scale
        
        s_qx =abs(mean_transform[3])+min_scale
        s_qy =abs(mean_transform[4])+min_scale
        s_qz =abs(mean_transform[5])+min_scale
        
        trans_norm = np.sqrt(s_tx*s_tx+s_ty*s_ty+s_tz*s_tz+s_qx*s_qx+s_qy*s_qy+s_qz*s_qz)
        
        normal_tx=s_tx/trans_norm
        normal_ty=s_ty/trans_norm
        normal_tz=s_tz/trans_norm 
        normal_qx=s_qx/trans_norm
        normal_qy=s_qy/trans_norm
        normal_qz=s_qz/trans_norm
        #0.09   + 1e-3

        self.search_size[3] = shape_scale * iou * normal_qx+min_scale
        self.search_size[4] = shape_scale * iou * normal_qy+min_scale
        self.search_size[5] = shape_scale * iou * normal_qz+min_scale
        self.search_size[0] = center_scale * iou * normal_tx+min_scale
        self.search_size[1] = center_scale * iou * normal_ty+min_scale
        self.search_size[2] = center_scale * iou * normal_tz+min_scale
        # print('self.search_size',self.search_size)

    
    def init_opt_params(self,box_3d,per_boxes_3d_R,per_boxes_3d_scores,verbose=False):
        '''
        box_3d: [N,6]
        per_boxes_3d_R: [N,3,3] 
        per_boxes_3d_scores: [N] 
        '''
        best_box = np.argmax(per_boxes_3d_scores) 

        mean_xyzlwh = np.zeros(6)
        box_center = box_3d[:,:3]
        mean_xyz = np.mean(box_center, axis=0) #[3]
        mean_xyzlwh[:3] = mean_xyz
        
        best_box_size = box_3d[best_box, 3:]
        sorted_indices = np.argsort(best_box_size)  # 
        index_0 = np.where(sorted_indices == 0)[0][0]
        index_1 = np.where(sorted_indices == 1)[0][0]
        index_2 = np.where(sorted_indices == 2)[0][0]
        get_indices = [index_0,index_1,index_2]
        
        B_sorted = np.sort(box_3d[:,3:], axis=1) #[N,3] s->l
        B_sorted = B_sorted[:, get_indices]
        if verbose:
            print('best_box_size',best_box_size)
            print("per_boxes_3d_scores",per_boxes_3d_scores)
            print("best_box",best_box)
            print("sorted_indices",sorted_indices)
            print('box_3d',box_3d)
            print('B_sorted',B_sorted)
        mean_xyzlwh[3:6] = np.mean(B_sorted,axis=0) #[3]
       

        mean_rot = per_boxes_3d_R[best_box] #[3,3]

        return mean_xyzlwh, mean_rot
    
    def init_opt_params_v2(self,box_3d,per_boxes_3d_R,per_boxes_3d_scores,verbose=False):
        '''
        box_3d: [N,6]
        per_boxes_3d_R: [N,3,3] 
        per_boxes_3d_scores: [N] 
        '''
        best_box = np.argmax(per_boxes_3d_scores) 

        mean_xyzlwh = np.zeros(6)
        box_center = box_3d[:,:3]
        mean_xyz = np.mean(box_center, axis=0) #[3]
        mean_xyzlwh[:3] = mean_xyz
        
        mean_xyzlwh[3:6] = np.mean(box_3d[:,3:],axis=0) #[3]
       
        mean_rot = per_boxes_3d_R[best_box] #[3,3]

        return mean_xyzlwh, mean_rot

    def init_opt_params_reliable(
        self,
        box_3d,
        per_boxes_3d_R,
        view_weights,
    ):
        mean_xyzlwh, mean_rot, _ = weighted_box_initialization(
            box_3d,
            per_boxes_3d_R,
            view_weights,
        )
        return mean_xyzlwh, mean_rot

    
    def boxfusion(self, all_pred_box, per_frame_box, box_manager, beta=0.9, verbose=False):
        N_box = len(all_pred_box)
        per_cam_pose = per_frame_box.cam_pose.cpu().numpy()
        per_boxes_3d = per_frame_box.pred_boxes_3d.tensor.cpu().numpy()
        per_boxes_3d_R = per_frame_box.get("pred_boxes_3d").R.cpu().numpy()
        per_boxes_3d_scores = per_frame_box.scores.cpu().numpy()

        per_boxes_2d = per_frame_box.pred_boxes.cpu().numpy()
        per_boxes_2d_cor = per_frame_box.projected_boxes.cpu().numpy()
        uncertainty_mode = self.boxer_uncertainty_cfg["mode"]
        final_uncertainty_mode = self.final_boxer_uncertainty_cfg["mode"]
        needs_boxer_metadata = (
            uncertainty_mode != "disabled"
            or final_uncertainty_mode != "disabled"
        )
        per_boxer_confidence = None
        per_boxer_applied = None
        if needs_boxer_metadata:
            required_fields = (
                "boxer_aleatoric_confidence",
                "boxer_geometry_applied",
            )
            missing_fields = [
                name for name in required_fields if not per_frame_box.has(name)
            ]
            if missing_fields:
                raise RuntimeError(
                    "Boxer uncertainty fusion is missing row-aligned fields: "
                    + ", ".join(missing_fields)
                )
            per_boxer_confidence = (
                per_frame_box.boxer_aleatoric_confidence.detach()
                .float()
                .cpu()
                .numpy()
            )
            per_boxer_applied = (
                per_frame_box.boxer_geometry_applied.detach()
                .bool()
                .cpu()
                .numpy()
            )
        for i in range(N_box):

            uncertainty_record = None
            final_uncertainty_recipe = None

            
            minimum_views = 3
            if self.reliable_view_cfg["enabled"]:
                minimum_views = max(
                    minimum_views,
                    self.reliable_view_cfg["min_views"],
                )
            if (
                len(box_manager.fusion_list[i]) < minimum_views
                or box_manager.check_if_fusion(
                    box_manager.fusion_list[i]
                )
            ):
                continue

            '''
            prepare the data used for fusion
            '''
            source_fusion_idx = list(box_manager.fusion_list[i])
            fusion_idx = np.asarray(source_fusion_idx, dtype=np.int64)

            if self.reliable_view_cfg["enabled"]:
                fusion_idx = stable_unique(fusion_idx)
                source_boxes = per_boxes_3d[fusion_idx]
                source_rotations = per_boxes_3d_R[fusion_idx]
                source_scores = per_boxes_3d_scores[fusion_idx]
                source_detector_boxes = per_boxes_2d[fusion_idx]
                source_corners = per_boxes_2d_cor[fusion_idx]
                source_poses = per_cam_pose[fusion_idx]
                source_boxer_confidence = None
                source_boxer_applied = None
                if needs_boxer_metadata:
                    source_boxer_confidence = per_boxer_confidence[fusion_idx]
                    source_boxer_applied = per_boxer_applied[fusion_idx]
                valid_views = valid_reliable_view_mask(
                    source_boxes,
                    source_rotations,
                    source_scores,
                    source_detector_boxes,
                    source_corners,
                    source_poses,
                )
                invalid_count = int(
                    valid_views.shape[0] - np.count_nonzero(valid_views)
                )
                self.reliable_view_stats["invalid_views"] += invalid_count
                if np.count_nonzero(valid_views) < minimum_views:
                    print(
                        f"reliable-view fusion {i} skipped: "
                        f"valid={int(np.count_nonzero(valid_views))}, "
                        f"required={minimum_views}, "
                        f"source={source_fusion_idx}"
                    )
                    continue
                fusion_idx = fusion_idx[valid_views]
                source_boxes = source_boxes[valid_views]
                source_scores = source_scores[valid_views]
                source_detector_boxes = source_detector_boxes[valid_views]
                source_corners = source_corners[valid_views]
                if needs_boxer_metadata:
                    source_boxer_confidence = source_boxer_confidence[
                        valid_views
                    ]
                    source_boxer_applied = source_boxer_applied[valid_views]
                base_selection = select_top_k_reliable_views(
                    source_boxes,
                    source_scores,
                    source_detector_boxes,
                    source_corners,
                    image_height=self.H,
                    image_width=self.W,
                    cfg=self.reliable_view_cfg,
                )
                if final_uncertainty_mode != "disabled":
                    final_uncertainty_recipe = {
                        "source_group": tuple(source_fusion_idx),
                        "candidate_indices": fusion_idx.copy(),
                        "base_selection": {
                            key: np.asarray(value).copy()
                            if isinstance(value, np.ndarray)
                            else value
                            for key, value in base_selection.items()
                        },
                        "projected_corners": source_corners.copy(),
                        "camera_poses": source_poses[valid_views].copy(),
                        "boxer_confidence": (
                            source_boxer_confidence.copy()
                        ),
                        "boxer_geometry_applied": (
                            source_boxer_applied.copy()
                        ),
                    }
                selection = base_selection
                if uncertainty_mode != "disabled":
                    uncertainty_started = time.perf_counter()
                    adjusted_selection = uncertainty_adjusted_selection(
                        base_selection,
                        source_boxer_confidence,
                        source_boxer_applied,
                        self.boxer_uncertainty_cfg,
                    )
                    uncertainty_runtime_ms = (
                        time.perf_counter() - uncertainty_started
                    ) * 1000.0
                    uncertainty_record = self.record_boxer_uncertainty_selection(
                        i,
                        fusion_idx,
                        base_selection,
                        adjusted_selection,
                        uncertainty_runtime_ms,
                    )
                    if uncertainty_mode == "active":
                        selection = adjusted_selection
                selected_local = selection["selected_indices"]
                fusion_idx = fusion_idx[selected_local]
                view_weights = selection["selected_weights"].astype(
                    np.float32
                )
                # Keep the weighted CUDA denominator on the same numerical
                # scale as the legacy observation count.
                view_weights = view_weights / max(
                    float(view_weights.mean()), 1e-6
                )
                self.record_reliable_view_selection(selection)
                print(
                    f"reliable-view fusion {i}: "
                    f"source={source_fusion_idx}, "
                    f"selected={fusion_idx.tolist()}, "
                    f"uncertainty={uncertainty_mode}, "
                    "weights="
                    f"{np.round(view_weights, 4).tolist()}"
                )
            else:
                view_weights = per_boxes_3d_scores[fusion_idx]

            num_of_boxes = len(fusion_idx)
            print(
                f"fusing {i} box, fusion list is ",
                fusion_idx.tolist(),
                "len:",
                num_of_boxes,
            )

            cam_poses = per_cam_pose[fusion_idx] #[N,4,4]
           
            box_3d = per_boxes_3d[fusion_idx] #[N,6] 

            corners_2d = per_boxes_2d_cor[fusion_idx] 

            if self.reliable_view_cfg["enabled"]:
                mean_xyzlwh, mean_rot = (
                    self.init_opt_params_reliable(
                        box_3d,
                        per_boxes_3d_R[fusion_idx],
                        view_weights,
                    )
                )
                scores_box = view_weights
            else:
                mean_xyzlwh, mean_rot = self.init_opt_params(
                    box_3d,
                    per_boxes_3d_R[fusion_idx],
                    per_boxes_3d_scores[fusion_idx],
                    verbose=False,
                )
                scores_box = per_boxes_3d_scores[fusion_idx]
            
            global_xyzlwh = mean_xyzlwh #initialize the parameters to be optimized
            

            self.init_searchsize()

            need_update = False
            previous_success = False
            fail_count = 0
        
            for n in range(self.fusion_iters):

                search_value = self.evaluate_iou(global_xyzlwh, 
                                                corners_2d,
                                                mean_rot, 
                                                scores_box,
                                                cam_poses,
                                                self.search_size,
                                                num_of_boxes,
                                                verbose=verbose,
                                                use_view_weights=self.reliable_view_cfg["enabled"])

                success,min_iou,mean_transform = self.cal_transform(search_value, 
                self.search_size)
                
                #update PST
                self.update_PST(min_iou,
                                mean_transform,
                                center_scale = self.center_scaling_coefficient,
                                shape_scale = self.shape_scaling_coefficient)
                                #scale=0.5) 
                
                if previous_success and success:
                    self.search_size[0] = beta*self.search_size[0]+(1-beta)*self.previous_search_size[0]
                    self.search_size[1] = beta*self.search_size[1]+(1-beta)*self.previous_search_size[1]
                    self.search_size[2] = beta*self.search_size[2]+(1-beta)*self.previous_search_size[2]
                    self.search_size[3] = beta*self.search_size[3]+(1-beta)*self.previous_search_size[3]
                    self.search_size[4] = beta*self.search_size[4]+(1-beta)*self.previous_search_size[4]
                    self.search_size[5] = beta*self.search_size[5]+(1-beta)*self.previous_search_size[5]

                #update global xyzlwh
                if success:
                    need_update = True
                    previous_success = True 
                    fail_count = 0

                    global_xyzlwh += mean_transform 
  
                    self.previous_search_size[0] = self.search_size[0]
                    self.previous_search_size[1] = self.search_size[1]
                    self.previous_search_size[2] = self.search_size[2]
                    self.previous_search_size[3] = self.search_size[3]
                    self.previous_search_size[4] = self.search_size[4]
                    self.previous_search_size[5] = self.search_size[5]

                else:
                    fail_count+=1
                    previous_success=False

                # shut down optimization if convergence
                if fail_count >= 3:
                    break

            if uncertainty_record is not None:
                uncertainty_record["optimization_updated"] = bool(
                    need_update
                )
                self.boxer_uncertainty_stats[
                    "optimization_updated_groups"
                ] += int(need_update)
                self.boxer_uncertainty_stats[
                    "active_updated_groups"
                ] += int(
                    need_update
                    and uncertainty_record["applied_to_fusion"]
                    and uncertainty_record["weights_changed"]
                )
                
            if need_update:
                # update tensor xyzlwh
                global_lwh = global_xyzlwh[3:]
                global_lwh[global_lwh < 0.01] = 0.01
                global_xyzlwh[3:] = global_lwh
                all_pred_box.pred_boxes_3d.tensor[i] = torch.from_numpy(global_xyzlwh).to(all_pred_box.pred_boxes_3d.tensor[i].device)
                if self.reliable_view_cfg["enabled"]:
                    all_pred_box.pred_boxes_3d.R[i] = torch.from_numpy(
                        mean_rot
                    ).to(all_pred_box.pred_boxes_3d.R[i].device)
                # update fusion flag
                box_manager.update_fusion_flag(i)
                box_manager.add_fusion_ind(source_fusion_idx)
            if final_uncertainty_recipe is not None:
                self.record_final_boxer_uncertainty_recipe(
                    global_box_index=i,
                    **final_uncertainty_recipe,
                )
