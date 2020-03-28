# Adapted from task-SI_scorer.py (Giovanni Da San Martino 2019, GPL license)

import importlib
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
scorer = importlib.import_module('tools.task-SI_scorer')

DEV_GS = '../data/si-dev-GS.txt'
TEST_GS = '../data/si-test-GS.txt'
# GS = DEV_GS  # Toggle!
GS = TEST_GS  # Toggle!


def compute_score_pr(submission_annotations, gold_annotations):
    prec_denominator = sum(
        [len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum(
        [len(annotations) for annotations in gold_annotations.values()])
    cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)

    for article_id in submission_annotations.keys():
        try:
            gold_data = gold_annotations[article_id]
        except KeyError:
            continue

        for j, sd in enumerate(submission_annotations[article_id]):
            sd_annotation_length = len(sd[1])
            for i, gd in enumerate(gold_data):
                try:
                    intersection = len(sd[1].intersection(gd[1]))
                    Spr_prec = intersection / sd_annotation_length
                except ZeroDivisionError:
                    Spr_prec = 0.0
                cumulative_Spr_prec += Spr_prec

                try:
                    gd_annotation_length = len(gd[1])
                    Spr_rec = intersection / gd_annotation_length
                except ZeroDivisionError:
                    Spr_rec = 0.0
                cumulative_Spr_rec += Spr_rec

    p, r, f1 = scorer.compute_prec_rec_f1(cumulative_Spr_prec,
                                          prec_denominator,
                                          cumulative_Spr_rec,
                                          rec_denominator)

    return f1, p, r


def eval(pred_file, gs_file=GS):
    techniques_names = ["propaganda"]
    submission_annotations = scorer.load_annotation_list_from_file(
        pred_file, techniques_names)

    # if GS == DEV_GS:
    #     # We don't have gold standard labels for the first article (730081389)
    #     submission_annotations.pop('730081389', None)

    gold_annotations = scorer.load_annotation_list_from_file(gs_file,
                                                             techniques_names)
    if not scorer.check_annotation_spans(submission_annotations, False):
        print("Error in submission file")
        sys.exit()
    scorer.check_annotation_spans(gold_annotations, True)
    p, r, f1 = compute_score_pr(submission_annotations, gold_annotations)
    print(p, r, f1)


if len(sys.argv) != 2:
    sys.stderr.write('Usage: ' + sys.argv[0] + ' PRED_FILE\n')
    sys.exit(1)
eval(sys.argv[1])
