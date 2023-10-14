import os
import pickle
import json
import numpy as np
from class_labels_utils import class_labels_n_ids


def gen_fusion_with_gt(data_path, scene_name, detic_exp, annotation_file, is_scannet):
    """This function generates the proposed_fusion_with_gt.pkl file from the proposed_fusion.pkl file"""
    # Parsing Path
    scene_path = os.path.join(data_path, "aligned_scans", scene_name)
    detic_path = os.path.join(scene_path, "detic_output", detic_exp)

    instances = []
    # Loading the ground-truth(gt) from the ScanNet dataset
    with open(os.path.join(scene_path, "{}_seg_by_indices.json".format(scene_name)), "rb") as f:
        instance_info = json.load(f)
    num_instances = len(instance_info)

    for entries in instance_info:
        gt_instance = dict()
        gt_instance["instance_id"] = num_instances
        num_instances += 1
        gt_instance["feature"] = None
        gt_instance["detections"] = None
        gt_instance["pt_indices"] = np.asarray(entries["indices"], dtype=np.int64)
        if is_scannet:
            if entries["label"] in class_labels_n_ids["scannet"]["VALID_CLASS_IDS_200"]:
                if class_labels_n_ids["scannet"]["CLASS_LABELS_200"][class_labels_n_ids["scannet"]["VALID_CLASS_IDS_200"].index(entries["label"])] not in class_labels_n_ids["scannet"]["BLACK_LIST"]:
                    gt_instance["top5_vocabs"] = [
                        class_labels_n_ids["scannet"]["CLASS_LABELS_200"][
                            class_labels_n_ids["scannet"]["VALID_CLASS_IDS_200"].index(entries["label"])
                        ]
                    ]
                    instances.append(gt_instance)

    # Saving the pkl file for the proposed fusion with gt (only once)
    with open(os.path.join(detic_path, "predictions", "proposed_fusion_gt.pkl"), "wb") as fp:
        pickle.dump(instances, fp)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # data_path = os.path.join(project_dir, "data/ScanNet")
    data_path = "/media/exx/T7 Shield"
    scene_name = "scene0011_00"
    detic_exp = "scan_net-0.3"
    annotation_file = "proposed_fusion_detic.pkl"
    for scene_name in os.listdir("/media/exx/T7 Shield/aligned_scans"):
        gen_fusion_with_gt(data_path, scene_name, detic_exp, annotation_file, is_scannet=True)
        pass
