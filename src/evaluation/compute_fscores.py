import numpy as np
import h5py
import os

from .evaluation_metrics import evaluate_summary
from .generate_summary import generate_summary

PATH = {
    'ovp': 'eccv16_dataset_ovp_google_pool5.h5',
    'summe': 'eccv16_dataset_summe_google_pool5.h5',
    'tvsum': 'eccv16_dataset_tvsum_google_pool5.h5',
    'youtube': 'eccv16_dataset_youtube_google_pool5.h5'
}


def f1_score(data, args):
    dataset_path = os.path.join(args.data, PATH[args.dataset])

    all_scores = []
    keys = list(data.keys())

    for video_name in keys:  # for each video inside that json file ...
        scores = data[video_name]  # read the importance scores from frames
        all_scores.append(scores)

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], \
                                                                   [], []
    with h5py.File(dataset_path, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]

            user_summary = np.array(hdf.get('video_' + video_index
                                            + '/user_summary'))
            sb = np.array(hdf.get('video_' + video_index + '/change_points'))
            n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
            positions = np.array(hdf.get('video_' + video_index + '/picks'))

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes,
                                     all_positions)

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, args.eval)
        all_f_scores.append(f_score)

    return np.mean(all_f_scores)
