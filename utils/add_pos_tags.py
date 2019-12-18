import numpy as np
import spacy
from sklearn.preprocessing import OneHotEncoder


def parse_input_file(infile, outfile):
    nlp = spacy.load("en_core_web_sm")

    with open(infile, encoding='utf8') as f_in:
        lines = f_in.readlines()
        lines.append('eof\teof\teof\teof\teof\teof\n')

    rows = []
    tokens = []
    prev_article = ''
    first_line = True
    comments = []
    labels = []
    lines_and_tags = []
    pos_tags = set()
    for line in lines:
        # Comments + header
        if line.startswith('#'):
            # f_out.write(line)
            comments.append(line)
            continue
        if first_line:
            comments.append('# POS tags = spacy\n')
            first_line = False
            labels = line.strip().split('\t')
            try:
                doc_idx = labels.index('document_id')
            except ValueError:
                doc_idx = 0
            try:
                word_idx = labels.index('token')
            except ValueError:
                word_idx = 4
            # labels.append('pos')
            # f_out.write('\t'.join(labels) + '\n')
            continue

        line = line[:-1]  # Remove \n
        fields = line.split('\t')
        article = fields[doc_idx]
        word = fields[word_idx]

        if article != prev_article:
            doc = nlp(' '.join(tokens))
            for tok, row in zip(doc, rows):
                lines_and_tags.append((row, tok.pos_))
                pos_tags.add(tok.pos_)
                # f_out.write(row)
                # f_out.write('\t')
                # f_out.write(tok.pos_)
                # f_out.write('\n')
            tokens = []
            rows = []
        tokens.append(word)
        rows.append(line)
        prev_article = article

    pos_tags = list(pos_tags)
    tags2onehot = {}
    M = len(pos_tags)
    for tag in pos_tags:
        tags2onehot[tag] = map(str, np.eye(N=1, M=M, k=pos_tags.index(tag),
                                           dtype=int).tolist())
    labels += ['pos_' + pos for pos in list(pos_tags)]
    with open(outfile, 'w', encoding='utf8') as f_out:
        for comment in comments:
            f_out.write(comment)
        f_out.write('\t'.join(labels))
        for line, tag in lines_and_tags:
            f_out.write(line)
            f_out.write('\t'.join(tags2onehot[tag]))
            f_out.write('\n')


if __name__ == "__main__":
    parse_input_file('../data/train-data-improved-sentiwordnet-arguingfull.tsv',
                     '../data/train-data-improved-sentiwordnet-arguingfull-pos.tsv')
    parse_input_file('../data/dev-improved-sentiwordnet-arguingfull.tsv',
                     '../data/dev-improved-sentiwordnet-arguingfull-pos.tsv')
