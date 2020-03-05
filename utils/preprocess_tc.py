"""
Preprocessing the datasets for task 2: technique classification.
"""
import os


TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TC_LABELS_FILE_DEV = "../datasets/dev-task-TC-template.out"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
DEV_DATA_FOLDER = "../datasets/dev-articles/"
TEST_DATA_FOLDER = "../datasets/test-articles/"


def get_spans_from_text(labels_file, raw_data_folder, file_to_write,
                        add_repetition_count=False, add_repetition_text=False):
    """
    Subtracts spans from raw texts and creates a new file
    which contains both labels and spans.

    :param labels_file: dir of the tab-separated file of form
        document_id    propaganda_label    beginning of span    end of span
    :param raw_data_folder: dir of folder with texts
    :param file_to_write: directory of the file to write
    """
    with open(labels_file, encoding='utf8') as f:
        table = f.readlines()
        table = [row.split() for row in table]

    open_doc_id = ""
    open_doc_txt = ""
    output_table = []

    header = ['document_id', 'label', 'span_start', 'span_end', 'text']
    if add_repetition_count:
        header.append('repetitions')
    output_table.append(header)

    for row in table:
        doc_id = row[0]
        from_id = int(row[2])        # idx of the beginning of the span
        to_id = int(row[3])          # idx of the end of the span

        # read the file if it's not opened yet
        if str(doc_id) != open_doc_id:
            with open(os.path.join(raw_data_folder,
                                   "article{}.txt".format(doc_id)),
                      encoding='utf8') as f:
                open_doc_txt = f.read()
                open_doc_id = doc_id

        span = open_doc_txt[from_id:to_id].strip()
        text = span.replace('\n', ' ')

        if add_repetition_count or add_repetition_text:
            n_reps = open_doc_txt.count(span)
            if add_repetition_text and n_reps > 1:
                text += ' ' + text
        if add_repetition_count:
            # -1 to count repetitions instead of occurrences
            output_table.append(row + [text] + [str(n_reps - 1)])
        else:
            output_table.append(row + [text])

    with open(file_to_write, 'w', encoding='utf8') as f:
        for row in output_table:
            f.write('\t'.join(row) + "\n")


def add_repetition_to_text(file_to_read, file_to_write):
    with open(file_to_read, "r", encoding='utf8') as fl:
        lines = fl.readlines()

    with open(file_to_write, "w", encoding='utf8') as fl:
        for line in lines:
            columns = line.strip().split("\t")
            if columns[1] == "Repetition":
                columns[4] = columns[4] + " " + columns[4]
            fl.write("\t".join(columns) + "\n")


if __name__ == '__main__':
    get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER,
                        "../data/tc-train.tsv", add_repetition_count=True)
    get_spans_from_text(TC_LABELS_FILE_DEV, DEV_DATA_FOLDER,
                        "../data/tc-dev.tsv", add_repetition_count=True)

    # For the 100% BERT model:
    get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER,
                        "../data/tc-train-repetition.tsv",
                        add_repetition_text=True)
    get_spans_from_text(TC_LABELS_FILE_DEV, DEV_DATA_FOLDER,
                        "../data/tc-dev-repetition.tsv",
                        add_repetition_text=True)
    # add_repetition_to_text("../data/tc-train.tsv",
    #                        "../data/tc-train-repetition.tsv")
