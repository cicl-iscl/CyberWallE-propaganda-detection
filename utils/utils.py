import os
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
DEV_DATA_FOLDER = "../datasets/dev-articles/"
SI_PREDICTIONS_FILE = '../data/dev_predictions_bio.tsv'
SI_SPANS_FILE = '../data/dev_predictions_spans.txt'


def get_si_dev_gs(tc_file='../datasets/dev-task-TC-template.out',
                  outfile='../data/dev-gs.txt'):
    articles2spans = {}
    with open(tc_file, encoding='utf8') as f_in:
        for line in f_in:
            fields = line[:-1].split('\t')
            article = fields[0]
            span_start = int(fields[2])
            span_end = int(fields[3])
            try:
                spans = articles2spans[article]
            except KeyError:
                spans = []
            spans.append((span_start, span_end))
            articles2spans[article] = spans

    rows = []
    articles = sorted([article for article in articles2spans])
    for article in articles:
        spans = articles2spans[article]
        spans.sort(key=lambda tup: tup[0])
        rows_in_article = []
        for span in spans:
            span = (article, span[0], span[1])
            if not rows_in_article:
                rows_in_article = [span]
                continue
            prev_span = rows_in_article[-1]
            if span[1] <= prev_span[2]:
                if span[2] <= prev_span[2]:
                    continue
                rows_in_article[-1] = (article, prev_span[1], span[2])
            else:
                rows_in_article.append(span)
        rows += rows_in_article

    with open(outfile, 'w', encoding='utf8') as f_out:
        for row in rows:
            f_out.write(row[0] + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n')


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


def annotate_text(raw_data_folder, labels_data_folder, file_to_write,
                  max_sent_len=35, improved_sent_splitting=True,
                  training=True):
    # max_sent_len = -1 ==> no sentence splitting
    if max_sent_len == -1:
        # the corresponding if-block can handle this
        improved_sent_splitting = True
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    if improved_sent_splitting:
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc',
                                        'ms', 'rep', 'u.s', 'feb', 'sen'])
        splitter = PunktSentenceTokenizer(punkt_param)
        splitter.PUNCTUATION = tuple(';:,.!?"')
    output_table = []
    file_counter = 0
    sent_no_total = 0

    print("Total number of files - {}".format(
        len(os.listdir(raw_data_folder))))

    # Reading all the files from the raw text directory
    article_file_names = [file_name for file_name in
                          os.listdir(raw_data_folder)
                          if file_name.endswith(".txt")]
    article_file_names.sort()

    for file_name in article_file_names:
        if training:
            label_file_name = file_name.replace(".txt", ".task2-TC.labels")
            print("raw_article: {}\tlabel_file: {}".format(file_name,
                                                           label_file_name))

            # Read the labels file with 4 columns of format
            # doc_id : label_of_span : idx_span_begin : idx_span_end
            with open(os.path.join(labels_data_folder, label_file_name),
                      encoding="utf-8") as file:
                rows = file.readlines()
                rows = [row.strip().split("\t") for row in rows
                        if len(row.split("\t")) == 4]

                # Saving mappings char_idx->labels into the dictionary
                char_idx2label = dict()
                for row in rows:
                    label = row[1]
                    idx_from = int(row[2])
                    idx_to = int(row[3])

                    for idx in range(idx_from, idx_to):
                        if idx not in char_idx2label.keys():
                            char_idx2label[idx] = []
                        char_idx2label[idx].append(label)
        else:
            print("raw_article: " + file_name)

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
                # Line breaks -> helps with headlines
                paragraphs = file_text.split('\n')
                for para in paragraphs:
                    para = para.strip()
                    sentences_raw = splitter.sentences_from_text(para)
                    for sent in sentences_raw:
                        sent = sent.strip()
                        tokens = tokenizer(sent)
                        if len(tokens) <= max_sent_len or max_sent_len == -1:
                            # No need to split the sentence!
                            if len(sent) == 0:
                                # Can happen when paragraphs are separated by
                                # several line breaks.
                                continue
                            sentences.append(sent)
                            continue

                        # TODO make this recursive

                        # Try splitting based on quotes.
                        quote_fragments, all_ok = punct_based_split_sent(
                            tokenizer, sent, max_sent_len, '"')
                        if all_ok:
                            sentences += quote_fragments
                            continue

                        # Other punctuation for splitting: ; :
                        for quote_frag in quote_fragments:
                            semicolon_fragments, all_ok =\
                                punct_based_split_sent(tokenizer, quote_frag,
                                                       max_sent_len, ';')
                            if all_ok:
                                sentences += semicolon_fragments
                                continue

                            for semicolon_frag in semicolon_fragments:
                                colon_fragments, all_ok =\
                                    punct_based_split_sent(tokenizer,
                                                           semicolon_frag,
                                                           max_sent_len, ':')
                                if all_ok:
                                    sentences += colon_fragments
                                    continue

                                # Commas:
                                for col_frag in colon_fragments:
                                    comma_fragments, all_ok =\
                                        punct_based_split_sent(tokenizer,
                                                               col_frag,
                                                               max_sent_len,
                                                               ',')
                                    if all_ok:
                                        sentences += comma_fragments
                                        continue

                                    # Last resort:
                                    # Split after max_sent_len tokens
                                    for comma_frag in comma_fragments:
                                        sentences += forcefully_split_sent(
                                            tokenizer, comma_frag,
                                            max_sent_len)
            else:
                # Cut long sentences into fragments that are (up to)
                # max_sent_len characters long
                # (the last fragment in a sentence might be shorter)
                file_text = file_text.replace('\n', ' ')
                sentences_raw = sent_tokenize(file_text)
                for sent in sentences_raw:
                    sentences += forcefully_split_sent(tokenizer, sent,
                                                       max_sent_len)

            i = 0
            for sent in sentences:
                sent = sent.strip()
                i = file_text.find(sent, i)
                max_idx = i + len(sent)

                if sent == '':
                    continue

                if improved_sent_splitting:
                    if len(sent.strip()) < 2:  # single char noise
                        continue

                sent_no_total += 1
                for token in tokenizer(sent):
                    token = str(token)
                    token_idx = file_text.find(token, i, max_idx)
                    i = token_idx + len(token)
                    output = [file_name.replace("article", "")
                                       .replace(".txt", ""),
                              str(sent_no_total),
                              str(token_idx),
                              str(i),
                              token]
                    if training:
                        # Check the label of the corresponding char_idx
                        label = char_idx2label.get(token_idx, ['None'])
                        output.append("|".join(label))
                    output_table.append(output)

        file_counter += 1
        print("Finished {} files\n".format(file_counter))

        with open(file_to_write, 'w', encoding="utf-8") as f:
            f.write('# max_sent_len=' + str(max_sent_len) +
                    ', improved_sent_splitting=' +
                    str(improved_sent_splitting) + '\n')
            f.write('document_id\tsent_id\ttoken_start\ttoken_end\ttoken')
            if training:
                f.write('\tlabel')
            f.write('\n')
            for row in output_table:
                f.write('\t'.join(row) + "\n")


# Helper method for annotate_text
def forcefully_split_sent(tokenizer, sent, max_sent_len):
    sentences = []
    tokens = tokenizer(sent)
    n_toks = len(tokens)
    if n_toks <= max_sent_len:
        if n_toks == 0:
            return []
        sentences.append(sent.strip())
        return sentences

    tok_idx = 0
    fragment_start = 0
    fragment_end = 0
    while tok_idx < n_toks:
        # This is so hacky >:(
        for token in tokens[tok_idx:tok_idx + max_sent_len]:
            fragment_end = sent.find(str(token),
                                     fragment_end) + len(token)
        sentences.append(sent[fragment_start:fragment_end]
                         .strip())
        tok_idx += max_sent_len
        fragment_start = fragment_end
    return sentences


# Helper method for annotate_text
def punct_based_split_sent(tokenizer, sent, max_sent_len, punct):
    if punct not in sent:
        return [sent], False
    sents = []
    tokens = tokenizer(sent)
    if len(tokens) <= max_sent_len:
        sents.append(sent)
        return sents, True
    # Try splitting along the punctuation mark.
    sent_fragments = sent.split(punct)
    n_frags = len(sent_fragments)
    prev_len = max_sent_len
    longest = 0
    for i, sent_fragment in enumerate(sent_fragments):
        if n_frags > 1 and i < n_frags - 1:
            sent_fragment += punct

        if len(sent_fragment.strip()) == 0:
            continue

        cur_len = len(tokenizer(sent_fragment.strip()))
        if cur_len > longest:
            longest = cur_len
        # We don't want to end up with a ton of very short sentences.
        if cur_len + prev_len <= max_sent_len:
            sents[-1] = sents[-1] + sent_fragment
            prev_len += cur_len
        else:
            sents.append(sent_fragment)
            prev_len = cur_len
    return sents, longest <= max_sent_len


# This relies on predictions ordered by article ID
def si_predictions_to_spans(si_predictions_file, span_file):
    with open(span_file, 'w', encoding='utf8') as outfile:
        with open(si_predictions_file, encoding='utf8') as infile:
            prev_label = 'O'
            prev_span_start = '-1'
            prev_span_end = '-1'
            prev_article = ''

            first_line = True
            for line in infile:

                # Comments + header
                if line.startswith('#'):
                    continue
                if first_line:
                    first_line = False
                    labels = line.strip().split('\t')
                    try:
                        article_idx = labels.index('document_id')
                    except ValueError:
                        article_idx = 0
                    try:
                        span_start_idx = labels.index('token_start')
                    except ValueError:
                        span_start_idx = 2
                    try:
                        span_end_idx = labels.index('token_end')
                    except ValueError:
                        span_end_idx = 3
                    try:
                        label_idx = labels.index('label')
                    except ValueError:
                        label_idx = -1
                    continue

                fields = line.strip().split('\t')
                article = fields[article_idx]
                span_start = fields[span_start_idx]
                span_end = fields[span_end_idx]
                label = fields[label_idx]

                prev_span_start = write_prediction(outfile, article, label,
                                                   span_start, span_end,
                                                   prev_article, prev_label,
                                                   prev_span_start,
                                                   prev_span_end)
                prev_article = article
                prev_label = label
                prev_span_end = span_end

        # Make sure we get the last prediction
        write_prediction(outfile,
                         article, label, span_start, span_end,
                         prev_article, prev_label, prev_span_start,
                         prev_span_end)


# Helper method for si_predictions_to_spans
def write_prediction(outfile,
                     article, label, span_start, span_end,
                     prev_article, prev_label, prev_span_start, prev_span_end):
    # Ending a span: I-O, B-O, I-B, B-B, new article
    if prev_label != 'O' and \
       (label != 'I' or prev_article != article):
        outfile.write(prev_article)
        outfile.write('\t')
        outfile.write(prev_span_start)
        outfile.write('\t')
        outfile.write(prev_span_end)
        outfile.write('\n')

    # Starting a new span: O-B, O-I, I-B, B-B, new article
    if label == 'B' or (label == 'I' and prev_label == 'O') \
            or prev_article != article:
        # Update the start of the current label span
        return span_start
    return prev_span_start


if __name__ == '__main__':
    LABELS_DATA_FOLDER = "../datasets/train-labels-task2-technique-classification/"
    # get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER, "../data/train-task2-TC-with-spans.labels")


    ###### BASELINE
    # annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
    #               "../data/train-data-with-sents-baseline-40.tsv",
    #               improved_sent_splitting=False, max_sent_len=40)
    # annotate_text(DEV_DATA_FOLDER, None,
    #               "../data/dev-baseline-40.tsv",
    #               improved_sent_splitting=False,
    #               training=False, max_sent_len=40)
    ######

    # annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
    #               "../data/train-data-sents-improved.tsv",
    #               improved_sent_splitting=True)

    # annotate_text(DEV_DATA_FOLDER, None,
    #               "../data/dev-improved.tsv",
    #               improved_sent_splitting=True,
    #               training=False)

    # annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
    #               "../data/train-data-fullsents.tsv",
    #               max_sent_len=-1)

    # annotate_text(DEV_DATA_FOLDER, None,
    #               "../data/dev-fullsents.tsv",
    #               max_sent_len=-1,
    #               training=False)

    get_si_dev_gs()