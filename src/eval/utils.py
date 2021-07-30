import os
import json

from ..datasets.voc import process_voc_xml

def ground_truth_to_coco_annotations(img_names, img_widths, img_heights, bboxes, labels, label_to_name, save_path):
    images = []
    annotations = []

    # loop over images
    for i, (name, width, height, img_boxes, img_labels) in enumerate(zip(img_names, img_widths, img_heights, bboxes, labels)):
        image_info = {
            "id": i,
            "width": width,
            "height": height,
            "file_name": name
        }
        images.append(image_info)

        # loop over detections in an image
        for box, label in zip(img_boxes, img_labels):
            image_ann = {
                "id": len(annotations),
                "image_id": i,
                "category_id": label,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            }
            annotations.append(image_ann)
    
    categories = [{"id": label, "name": name} for label, name in label_to_name.items()]

    ann = {
        "info": None,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    
    with open(save_path, "w") as f:
        json.dump(ann, f)
    
    return ann

def voc_to_coco_annotations(data_dir, split, name_to_label, save_path):
    img_list = os.path.join(data_dir, "ImageSets", "Main", f"{split}.txt")
    with open(img_list, "r") as f:
        img_names = [x.rstrip() for x in f]
    
    img_widths = []
    img_heights = []
    labels = []
    bboxes = []

    ann_dir = os.path.join(data_dir, "Annotations")
    for img_name in img_names:
        ann_file = os.path.join(ann_dir, f"{img_name}.xml")

        # original voc is x1y1x2y2
        annotation = process_voc_xml(ann_file, original_bboxes=True)
        img_widths.append(annotation["img_width"])
        img_heights.append(annotation["img_height"])
        img_bboxes = annotation["bboxes"]
        names = annotation["names"]

        # convert voc x1y1x2y2 to coco xywh
        for box in img_bboxes:
            box[2] -= box[0]
            box[3] -= box[1]

        img_labels = [name_to_label[x] for x in names]

        labels.append(img_labels)
        bboxes.append(img_bboxes)
    
    img_names = [f"{x}.jpg" for x in img_names]
    label_to_name = {v: k for k,v in name_to_label.items()}
    ann = ground_truth_to_coco_annotations(img_names, img_widths, img_heights, bboxes, labels, label_to_name, save_path)
    return ann

def detections_to_coco_results(image_ids, bboxes, labels, scores, save_path, score_threshold=0):
    results = []

    for img_id, img_bboxes, img_labels, img_scores in zip(image_ids, bboxes, labels, scores):
        
        for box, label, score in zip(img_bboxes, img_labels, img_scores):
            if score < score_threshold:
                continue

            item = {
                "image_id": img_id,
                "category_id": int(label),
                "bbox": box,
                "score": score
            }
            results.append(item)

    with open(save_path, "w") as f:
        json.dump(results, f)

    return results
