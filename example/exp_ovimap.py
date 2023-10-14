"""Example on directly using Open-Vocabulary Instance Map (ovimap)"""
import os
import cv2
import numpy as np
from ovsg.env.ovimap.ovimap import OVIMapDetic


if __name__ == "__main__":
    # test case 1
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = f"{root_dir}/test_data"
    scene_name = "scene0645_00"  # "scene0645_00", "cus_scene0001_00"
    detic_exp = "scannet200-0.3"  # "imagenet21k-0.3", "scannet200-0.3"
    annotation_file = "proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl"
    geometry_file, anno_file, info_file, color_img_path, detic_path = OVIMapDetic.parse_path(
        data_path, scene_name, annotation_file, "custom_scannet", detic_exp=detic_exp
    )
    intrinsic = np.array([[1000, 0, 399.5], [0, 1000, 499.5]])  # o3d requires .5
    img_height, img_width = 800, 1000
    ovimap = OVIMapDetic(
        geometry_file,
        info_file,
        img_height,
        img_width,
        intrinsic,
        None,
        device="cpu",
        detic_path=detic_path,
        anno_file=anno_file,
        color_img_path=color_img_path,
    )

    queries = ["book", "bed", "sofa", "chair", "towel"]

    for query in queries:
        queried_instance = ovimap.query(query, top_k=3)
        for i, inst in enumerate(queried_instance):
            ovimap.mark(inst)
        img = ovimap.render(show_inst_img=True)
        cv2.imshow(f"query-{query}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ovimap.visualize_3d(show_bbox=True, show_origin=True)
        ovimap.clear_mark()
