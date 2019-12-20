import importlib
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
scorer = importlib.import_module('tools.task-SI_scorer')

DEV_GS = '../data/dev-gs.txt'


def eval(pred_file, gs_file=DEV_GS):
    techniques_names = ["propaganda"]
    submission_annotations = scorer.load_annotation_list_from_file(
        pred_file, techniques_names)
    gold_annotations = scorer.load_annotation_list_from_file(gs_file,
                                                             techniques_names)
    scorer.check_data_file_lists(submission_annotations, gold_annotations)
    if not scorer.check_annotation_spans(submission_annotations, False):
        print("Error in submission file")
        sys.exit()
    scorer.check_annotation_spans(gold_annotations, True)
    f1 = scorer.compute_score_pr(submission_annotations,
                                 gold_annotations, techniques_names,
                                 prop_vs_non_propaganda=True,
                                 per_article_evaluation=False,
                                 output_for_script=False)
    print(f1)


if len(sys.argv) != 2:
    sys.stderr.write('Usage: ' + sys.argv[0] + ' PRED_FILE\n')
    sys.exit(1)
eval(sys.argv[1])
