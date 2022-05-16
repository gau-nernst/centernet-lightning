import tempfile
import os
import sys

import trackeval
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity

def evaluate_mot_tracking_sequence(pred_bboxes, pred_track_ids, target_bboxes, target_track_ids):
    """bboxes in xywh format

    This will create the following folder structure
    ```
    temp_dir (from tempfile.TemporaryDirectory())
    |—— gt
    |   |—— seqmap.txt          # contains "sequence_0"
    |   |—— sequence_0
    |   |   |—— seqinfo.ini     # sequence info
    |   |   |—— gt
    |   |   |   |——gt.txt       # contains target tracks
    |—— trackers
    |   |—— tracker_0
    |   |   |—— sequence_0.txt  # contains predicted tracks
    ```
    """
    sequence_name = "sequence_0"
    tracker_name = "tracker_0"

    with tempfile.TemporaryDirectory() as temp_dir:
        # create ground truth folder
        gt_folder = os.path.join(temp_dir, "gt")            # temp/gt
        seq_dir = os.path.join(gt_folder, sequence_name)    # temp/gt/sequence_0
        seq_gt_dir = os.path.join(seq_dir, "gt")            # temp/gt/sequence_0/gt
        os.makedirs(seq_gt_dir)

        # create sequence map file
        seqmap_file = os.path.join(gt_folder, "seqmap.txt")
        with open(seqmap_file, "w") as f:
            f.write(f"name\n{sequence_name}")

        # write sequence info
        seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
        with open(seqinfo_path, "w") as f:
            content = f"""
                [Sequence]\n
                name={sequence_name}\n
                imDir=img1\n
                seqLength={len(target_track_ids)}\n
                imWidth=1\n
                imHeight=1\n
                imExt=.jpg
            """
            f.write(content)

        # write target tracks to file
        gt_path = os.path.join(seq_gt_dir, "gt.txt")
        with open(gt_path, "w") as f:
            for i, (frame_bboxes, frame_track_ids) in enumerate(zip(target_bboxes, target_track_ids)):
                for box, track_id in zip(frame_bboxes, frame_track_ids):
                    # MOT Challenge uses 1-based index
                    line = f"{i+1},{track_id+1},{box[0]+1},{box[1]+1},{box[2]},{box[3]},1,1,1\n"
                    f.write(line)

        # create trackers and prediction folder
        trackers_folder = os.path.join(temp_dir, "trackers") 
        os.makedirs(trackers_folder)
        tracker_dir = os.path.join(trackers_folder, tracker_name, "data")
        os.makedirs(tracker_dir)

        # write predicted tracks to file
        pred_path = os.path.join(tracker_dir, f"{sequence_name}.txt")
        with open(pred_path, "w") as f:
            for i, (frame_bboxes, frame_track_ids) in enumerate(zip(pred_bboxes, pred_track_ids)):
                for box, track_id in zip(frame_bboxes, frame_track_ids):
                    # MOT Challenge uses 1-based index
                    line = f"{i+1},{track_id+1},{box[0]+1},{box[1]+1},{box[2]},{box[3]},1,-1,-1,-1\n"
                    f.write(line)

        metrics = evaluate_mot_tracking_from_file(gt_folder, trackers_folder, trackers_to_eval=[tracker_name], seqmap_file=seqmap_file, skip_split_fol=True)

    metrics = metrics[tracker_name][sequence_name]
    metrics = {"HOTA": metrics["HOTA"].mean(), "MOTA": metrics["MOTA"], "IDF1": metrics["IDF1"]}
    return metrics

# https://github.com/JonathonLuiten/TrackEval/blob/master/scripts/run_mot_challenge.py
# NOTE: when there are too many tracks, TrackEval will use too much memory and the computer will freeze
def evaluate_mot_tracking_from_file(gt_folder, trackers_folder, use_parallel=True, num_parallel_cores=4, trackers_to_eval=None, seqmap_file=None, skip_split_fol=False, save_results=False):
    # save and redirect print() to null
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/eval.py
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = use_parallel
    eval_config["NUM_PARALLEL_CORES"] = num_parallel_cores
    for key in ("OUTPUT_SUMMARY", "OUTPUT_EMPTY_CLASSES", "OUTPUT_DETAILED", "PLOT_CURVES"):
        eval_config[key] = save_results
    evaluator = trackeval.Evaluator(eval_config)

    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_folder
    dataset_config["TRACKERS_FOLDER"] = trackers_folder
    dataset_config["TRACKERS_TO_EVAL"] = trackers_to_eval
    dataset_config["SEQMAP_FILE"] = seqmap_file
    dataset_config["SKIP_SPLIT_FOL"] = skip_split_fol
    dataset_list = [MotChallenge2DBox(dataset_config)]

    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py
    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py
    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py
    metrics_list = [HOTA(), CLEAR(), Identity()]
    
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    results = results["MotChallenge2DBox"]

    sys.stdout = old_stdout     # restore print()

    # results = {
    #     "tracker_1": {
    #         "sequence_1": {
    #             "pedestrian": {
    #                 "HOTA": {...},
    #                 "CLEAR": {...},
    #                 "Identity": {...}
    #             }
    #         },
    #         "sequence_2": {...}
    #     },
    #     "tracker_2": {...}
    # }
    for tracker, tracker_result in results.items():
        for seq, seq_result in tracker_result.items():
            seq_result = seq_result["pedestrian"]                                           # remove "pedestrian" key
            seq_result = {k:v for metric in seq_result.values() for k,v in metric.items()}  # unroll nested dict
            results[tracker][seq] = seq_result
    
    return results
