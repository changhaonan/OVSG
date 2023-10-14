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
    with open(os.path.join(scene_path, "annotation_info.pkl".format(scene_name)), "rb") as f:
        instance_info = pickle.load(f)
    with open(os.path.join(scene_path, "annotation_info_extra.pkl".format(scene_name)), "rb") as f:
        instance_info2 = pickle.load(f)
    num_instances = 0

    for entries in instance_info:
        gt_instance = dict()
        gt_instance["instance_id"] = num_instances
        num_instances += 1
        gt_instance["feature"] = None
        gt_instance["detections"] = None
        gt_instance["pt_indices"] = np.asarray(entries["pt_indices"], dtype=np.int64)
        if is_scannet:
            class_name = ''.join(char for char in entries["top5_vocabs"][0] if not char.isdigit()).lower()
            if class_name == "trashcan":
                 class_name = "trash can"
            if class_name in class_labels_n_ids["scannet"]["CLASS_LABELS_200"]:
                # if class_labels_n_ids["scannet"]["CLASS_LABELS_200"][class_labels_n_ids["scannet"]["VALID_CLASS_IDS_200"].index(entries["label"])] not in class_labels_n_ids["scannet"]["BLACK_LIST"]:
                    # print(class_name)
                    gt_instance["top5_vocabs"] = [
                         class_name
                        ]
                    instances.append(gt_instance)


    for entries in instance_info2:
        gt_instance = dict()
        gt_instance["instance_id"] = num_instances
        
        gt_instance["feature"] = None
        gt_instance["detections"] = None
        gt_instance["pt_indices"] = np.asarray(entries["pt_indices"], dtype=np.int64)
        if is_scannet:
            class_name = ''.join(char for char in entries["top5_vocabs"][0] if not char.isdigit()).lower()
            if class_name == "mouse" or class_name == "keyboard":
                if class_name in class_labels_n_ids["scannet"]["CLASS_LABELS_200"]:
                # if class_labels_n_ids["scannet"]["CLASS_LABELS_200"][class_labels_n_ids["scannet"]["VALID_CLASS_IDS_200"].index(entries["label"])] not in class_labels_n_ids["scannet"]["BLACK_LIST"]:
                    # print(class_name)
                    gt_instance["top5_vocabs"] = [
                         class_name
                        ]
                    instances.append(gt_instance) 
                    num_instances += 1   
    # Saving the pkl file for the proposed fusion with gt (only once)
    with open(os.path.join(detic_path, "predictions", "proposed_fusion_gt.pkl"), "wb") as fp:
        pickle.dump(instances, fp)
    pass


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # data_path = os.path.join(project_dir, "data/ScanNet")
    data_path = "/media/exx/T7 Shield"
    scene_name = "cus_scene0000_01"
    detic_exp = "scan_net-0.3"
    annotation_file = "proposed_fusion.pkl"
    # for scene_name in os.listdir("/media/exx/T7 Shield/aligned_scans"):
    gen_fusion_with_gt(data_path, scene_name, detic_exp, annotation_file, is_scannet=True)
    pass
