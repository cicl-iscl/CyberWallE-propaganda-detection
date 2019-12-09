# Changes labels from category-specific labels to BIO-style labels.
import sys


def overlap(l1, l2):
    if l1 == l2:
        return True
    if l1 in l2:
        return True
    if l2 in l1:
        return True
    return False


def labels2bio(span_file, bio_file):
    with open(span_file, encoding='utf8') as infile:
        with open(bio_file, 'w', encoding='utf8') as outfile:
            prev_label = 'None'
            prev_article = '-1'
            first_line = True
            for line in infile:

                # Comments + header
                if line.startswith('#'):
                    outfile.write(line)
                    continue
                if first_line:
                    first_line = False
                    outfile.write(line)
                    labels = line.strip().split('\t')
                    try:
                        doc_idx = labels.index('document_id')
                    except ValueError:
                        doc_idx = 0
                    try:
                        label_idx = labels.index('label')
                    except ValueError:
                        label_idx = len(labels) + 1
                    continue

                fields = line.strip().split('\t')
                article = fields[doc_idx]
                label = fields[label_idx]

                if label == 'None':
                    bio_label = 'O'
                elif overlap(prev_label, label) and prev_article == article:
                    bio_label = 'I'
                else:
                    bio_label = 'B'
                prev_label = label
                prev_article = article

                fields[label_idx] = bio_label
                outfile.write('\t'.join(fields))
                outfile.write('\n')


if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     sys.stderr.write('Usage:', sys.argv[0] + ' INFILE OUTFILE\n')
    #     sys.exit(1)

    # labels2bio(sys.argv[1], sys.argv[2])

    labels2bio('../data/train-data-sents-improved.tsv',
               '../data/train-data-improved.tsv')
