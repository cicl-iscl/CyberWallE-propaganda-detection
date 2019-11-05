import os
import nltk

TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"

def get_spans_from_text(labels_file, raw_data_folder, file_to_write):
    """
    Subtract spans from raw texts and create a new file
    which contains both labels and spans.
    
    :param labels_file: dir of the tab-separated file of form 
                        document_id - propaganda_label - beginning of span - end of span 
    :param raw_data_folder: dir of folder with texts
    :param file_to_write: directory of the file to write
    """
    with open(labels_file) as f:
        table = f.readlines()
        table = [row.split() for row in table]

    open_doc_id = ""
    open_doc_txt = ""
    output_table = []

    for row in table:
        doc_id = row[0]
        from_id = int(row[2])        # idx of the beginning of the span
        to_id = int(row[3])          # idx of the end of the span

        # read the file if it's not opened yet
        if str(doc_id) != open_doc_id:
            with open(os.path.join(raw_data_folder, "article{}.txt".format(doc_id))) as f:
                open_doc_txt = f.read()
                open_doc_id = doc_id

        span = open_doc_txt[from_id:to_id].strip()
        output_table.append(row + [span])

    with open(file_to_write, 'w') as f:
        for row in output_table:
            f.write('\t'.join(row) + "\n")


def annotate_text(labels_file, raw_data_folder, file_to_write):

    # Reading data from the file with labels
    with open(labels_file) as f:
        table = f.readlines()
        table = [row.strip().split() for row in table]

    # Saving mappings document_id->char_idx->labels into dictionaries
    doc2char_idx = dict()
    for row in table:
        doc_id, label, idx_from, idx_to = row[0], row[1], int(row[2]), int(row[3])

        if doc_id not in doc2char_idx.keys():
            doc2char_idx[doc_id] = dict()

        for idx in range(idx_from, idx_to):
            if idx not in doc2char_idx[doc_id].keys():
                doc2char_idx[doc_id][idx] = []

            doc2char_idx[doc_id][idx].append(label)

    output_table = []

    # Reading all the files from the raw text directory

    print("Total number of files - {}".format(len(os.listdir(raw_data_folder))))

    file_counter = 0
    lost_files = []

    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".txt"):
            doc_id = filename.replace("article", "").replace(".txt", "")
            print(doc_id)

            if doc_id in doc2char_idx.keys():
                with open(os.path.join(raw_data_folder, filename)) as f:
                    file_text = f.read()

                    tokens = nltk.word_tokenize(file_text)

                    doc_length = len(file_text)
                    char_idx = 0

                    while char_idx < doc_length:
                        count_updated = False

                        for token in tokens:
                            if file_text[char_idx:].startswith(token):

                                if char_idx in doc2char_idx[doc_id].keys():
                                    label = doc2char_idx[doc_id][char_idx]
                                else:
                                    label = ["None"]

                                output_table.append([doc_id, str(char_idx), str(char_idx+len(token)), token, "|".join(label)])
                                char_idx += len(token)
                                count_updated = True
                                tokens.pop(0)
                                break

                        if not count_updated:
                            char_idx += 1
            else:
                lost_files += doc_id

        file_counter += 1
        print("Finished {} files".format(file_counter))

        with open(file_to_write, 'w') as f:
            for row in output_table:
                f.write('\t'.join(row) + "\n")

    print("Number of lost files - {}".format(len(lost_files)))
    print(lost_files)

    # TODO: solve an issue with lost files. 126 files were not processed

    # with open(os.path.join(raw_data_folder), encoding="utf-8") as f:
    #     file_text = " ".join(f.read().split())
    #
    #     tokens = nltk.word_tokenize(file_text)
    #
    #     for token in tokens:
    #         print(token)
    #
    #     curr_char = 0
    #
    #     while len(tokens) > 0:
    #         token = tokens[0]
    #         print("Token ", token)
    #         print("File Text ", file_text[:20])
    #         if file_text.startswith(token):
    #
    #             if curr_char in doc2char_idx[doc_id].keys():
    #                 label = doc2char_idx[doc_id][curr_char]
    #             else:
    #                 label = "None"
    #
    #             print([curr_char, curr_char+len(token), token, label])
    #             curr_char += len(token)
    #             file_text = file_text[len(token):]
    #             tokens.pop(0)
    #         else:
    #             curr_char += 1
    #             file_text = file_text[1:]


if __name__ == '__main__':
    TEST = "../datasets/train-articles/article111111111.txt"
    # get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER, "../data/train-task2-TC-with-spans.labels")
    annotate_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER, "../data/train-data.tsv")