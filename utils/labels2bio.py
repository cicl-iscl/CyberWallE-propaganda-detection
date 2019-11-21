# Changes the labels in the training file from category-specific labels
# to BIO-style labels.


def overlap(l1, l2):
    if l1 == l2:
        return True
    if l1 in l2:
        return True
    if l2 in l1:
        return True
    return False


def labels2bio(span_file, bio_file, include_sent_number=True):
    with open(span_file, encoding='utf8') as infile:
        with open(bio_file, 'w', encoding='utf8') as outfile:
            prev_label = 'None'
            for line in infile:
                fields = line.strip().split('\t')
                label = fields[5]
                outfile.write(fields[0])
                outfile.write('\t')
                if include_sent_number:
                    outfile.write(fields[1])
                    outfile.write('\t')
                outfile.write(fields[2])
                outfile.write('\t')
                outfile.write(fields[3])
                outfile.write('\t')
                outfile.write(fields[4])
                outfile.write('\t')
                if label == 'None':
                    outfile.write('O')
                elif overlap(prev_label, label):
                    outfile.write('I')
                else:
                    outfile.write('B')
                prev_label = label
                outfile.write('\n')


if __name__ == '__main__':

    # We could get the arguments via sys.argv, but we don't need to run this
    # method a lot...

    labels2bio('../data/train-data-with-sents-baseline.tsv',
               '../data/train-data-bio-baseline.tsv')
    # labels2bio('../data/train-data-with-sents.tsv',
    #            '../data/si-sample-predictions.tsv', False)
