# Changes labels from category-specific labels to BIO-style labels.
import sys

PATH_TO_SENT = '../data/sentiment/SentiWords_1.1.txt'


def parse_lexicon(lexicon_file):
    lex = dict()
    prev_word = ''
    score = 0
    n_entries = 0

    lines = []
    with open(lexicon_file, encoding='utf8') as f:
        lines = f.readlines()
        lines += ['end-of-file\t0']

    for line in lines:
        if line.startswith('#'):
            # comment
            continue
        fields = line.split('\t')
        # word#pos    value
        # TODO update if we include POS info
        word = fields[0].split('#')[0]
        value = float(fields[1])
        if word == prev_word:
            score += value
            n_entries += 1
        else:
            if n_entries > 0:
                lex[word] = score / n_entries
            n_entries = 0
            score = 0
        prev_word = word

    return lex


def annotate(lex, infile, outfile, word_idx=4):
    with open(infile, encoding='utf8') as f_in:
        with open(outfile, 'w', encoding='utf8') as f_out:
            for line in f_in:
                line = line[:-1]  # Remove \n
                word = line.split('\t')[word_idx].lower()
                value = lex.get(word, 0.0)
                f_out.write(line + '\t' + str(value) + '\n')


if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     sys.stderr.write('Usage:', sys.argv[0] + ' INFILE OUTFILE\n')
    #     sys.exit(1)

    lex = parse_lexicon(PATH_TO_SENT)
    annotate(lex, '../data/train-data-bio-improved.tsv',
             '../data/train-data-bio-improved-sentiment.tsv')
    annotate(lex, '../data/dev-improved.tsv',
             '../data/dev-improved-sentiment.tsv')
