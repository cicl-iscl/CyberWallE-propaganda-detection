import os
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
SI_PREDICTIONS_FILE = '../data/si-sample-predictions.tsv'
SI_SPANS_FILE = '../data/si-sample-spans.tsv'


def get_spans_from_text(labels_file, raw_data_folder, file_to_write):
    """
    Subtract spans from raw texts and create a new file
    which contains both labels and spans.
    
    :param labels_file: dir of the tab-separated file of form 
                        document_id - propaganda_label - beginning of span - end of span 
    :param raw_data_folder: dir of folder with texts
    :param file_to_write: directory of the file to write
    """
    with open(labels_file, encoding='utf8') as f:
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
            with open(os.path.join(raw_data_folder, "article{}.txt".format(doc_id)), encoding='utf8') as f:
                open_doc_txt = f.read()
                open_doc_id = doc_id

        span = open_doc_txt[from_id:to_id].strip().replace('\n', ' ')
        output_table.append(row + [span])

    with open(file_to_write, 'w') as f:
        for row in output_table:
            f.write('\t'.join(row) + "\n")


def annotate_text(raw_data_folder, labels_data_folder, file_to_write, max_sent_len=35, improved_sent_splitting=True):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    if improved_sent_splitting:
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'ms', 'rep', 'u.s', 'feb', 'sen'])
        splitter = PunktSentenceTokenizer(punkt_param)
        splitter.PUNCTUATION = tuple(';:,.!?"')
    output_table = []
    file_counter = 0
    sent_no_total = 0

    print("Total number of files - {}".format(len(os.listdir(raw_data_folder))))

    # Reading all the files from the raw text directory
    article_file_names = [file_name for file_name in os.listdir(raw_data_folder)
                          if file_name.endswith(".txt")]
    article_file_names.sort()

    for file_name in article_file_names:
        label_file_name = file_name.replace(".txt", ".task2-TC.labels")

        print("raw_article: {}\tlabel_file: {}".format(file_name, label_file_name))

        # Read the labels file with 4 columns of format
        # doc_id : label_of_span : idx_span_begin : idx_span_end
        with open(os.path.join(labels_data_folder, label_file_name), encoding="utf-8") as file:
            rows = file.readlines()
            rows = [row.strip().split("\t") for row in rows if len(row.split("\t")) == 4]

            #Saving mappings char_idx->labels into the dictionary
            char_idx2label = dict()
            for row in rows:
                _, label, idx_from, idx_to = row[0], row[1], int(row[2]), int(row[3])

                for idx in range(idx_from, idx_to):
                    if idx not in char_idx2label.keys():
                        char_idx2label[idx] = []
                    char_idx2label[idx].append(label)

        # Read the article and process the text
        with open(os.path.join(raw_data_folder, file_name),
                  encoding="utf-8") as file:
            file_text = file.readlines()
            # Keep linebreaks for better sentence splitting
            file_text = ''.join([line for line in file_text])

            # Normalizing punctuation marks to help the tokenizer.
            file_text = file_text.replace('“', '"').replace('”', '"')
            file_text = file_text.replace("’", "'").replace("‘", "'")

            sentences = []
            if improved_sent_splitting:
                paragraphs = file_text.split('\n')
                for para in paragraphs:
                    para = para.strip()
                    sentences_raw = splitter.sentences_from_text(para)
                    for sent in sentences_raw:
                        sent = sent.strip()
                        tokens = tokenizer(sent)
                        if len(tokens) <= max_sent_len:
                            if len(sent) == 0:
                                continue
                            sentences.append(sent)
                            continue

                        # TODO make sure this actually reduces the sentences to 35 tokens max
                        if '"' in sent:
                            for i, sent_fragment in enumerate(sent.split('"')):
                                if i % 2 == 1:  # Inside a quote
                                    sent_fragment = '"' + sent_fragment + '"'
                                else:
                                    sent_fragment = sent_fragment.strip()
                                    if len(sent_fragment) == 0:
                                        continue
                                sentences.append(sent_fragment)
                        else:
                            # TODO
                            sentences.append(sent.strip())
            else:
                # Cut long sentences into fragments that are (up to)
                # max_sent_len characters long
                # (the last fragment in a sentence might be shorter)
                file_text = file_text.replace('\n', ' ')
                sentences_raw = sent_tokenize(file_text)
                for sent in sentences_raw:
                    tokens = tokenizer(sent)
                    n_toks = len(tokens)
                    if n_toks <= max_sent_len:
                        if n_toks == 0:
                            continue
                        sentences.append(sent.strip())
                        continue
                    tok_idx = 0
                    fragment_start = 0
                    fragment_end = 0
                    while tok_idx < n_toks:
                        # This is so hacky >:(
                        sent_fragment = ''
                        for token in tokens[tok_idx:tok_idx + max_sent_len]:
                            fragment_end = sent.find(str(token),
                                                     fragment_end) + len(token)
                        sentences.append(sent[fragment_start:fragment_end].strip())
                        tok_idx += max_sent_len
                        fragment_start = fragment_end

            try:
                sentences.remove('')
            except ValueError:
                pass  # No empty snippets in the first place

            sent_indices = []
            idx = 0
            for sent in sentences:
                idx = file_text.find(sent, idx)
                sent_indices.append(idx)
                idx += len(sent)
            sent_indices.append(len(file_text))

            sent_no = -1
            for i in sent_indices[:-1]:
                sent_no += 1
                sent_no_total += 1
                max_idx = sent_indices[sent_no + 1]  # start of next sent
                tokens = tokenizer(sentences[sent_no].strip())
                for token in tokens:
                    token = str(token)
                    token_idx = file_text.find(token, i, max_idx)
                    # Check the label of the corresponding char_idx
                    label = char_idx2label.get(token_idx, ['None'])
                    i = token_idx + len(token)
                    output_table.append([file_name.replace("article", "")
                                                  .replace(".txt", ""),
                                         str(sent_no_total),
                                         str(token_idx),
                                         str(i),
                                         token,
                                         "|".join(label)])

        file_counter += 1
        print("Finished {} files\n".format(file_counter))

        with open(file_to_write, 'w', encoding="utf-8") as f:
            for row in output_table:
                f.write('\t'.join(row) + "\n")


# TODO deal with spans contained in other spans
# This relies on predictions ordered by article ID
def si_predictions_to_spans(si_predictions_file, span_file):
    # Make sure we get the last prediction at the end of the line-reading loop
    # by adding a dummy line:
    lines = []
    with open(si_predictions_file, encoding='utf8') as infile:
        lines = infile.readlines()
        lines += ['end-of-file\t-1\t-1\tdummy\tO\n']

    with open(span_file, 'w', encoding='utf8') as outfile:
        prev_label = 'O'
        prev_span_start = -1
        prev_span_end = -1
        prev_article = ''
        for line in lines:
            fields = line.strip().split('\t')
            article = fields[0]
            span_start = fields[1]
            span_end = fields[2]
            # fields[3] is the word itself
            label = fields[4]

            # Ending a span: I-O, B-O, I-B, B-B
            if prev_label != 'O' and \
               (label != 'I' or prev_article != article):
                outfile.write(prev_article)
                outfile.write('\t')
                outfile.write(prev_span_start)
                outfile.write('\t')
                outfile.write(prev_span_end)
                outfile.write('\n')

            # Starting a new span: O-B, O-I, I-B, B-B
            if label == 'B' or (label == 'I' and prev_label == 'O'):
                prev_span_start = span_start

            prev_article = article
            prev_label = label
            prev_span_end = span_end


if __name__ == '__main__':
    LABELS_DATA_FOLDER = "../datasets/train-labels-task2-technique-classification/"
    # get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER, "../data/train-task2-TC-with-spans.labels")

    # annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
    #               "../data/train-data-with-sents-baseline.tsv",
    #               improved_sent_splitting=False)    # an

    annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
                  "../data/train-data-with-sents-improved.tsv",
                  improved_sent_splitting=True)
    si_predictions_to_spans(SI_PREDICTIONS_FILE, SI_SPANS_FILE)
