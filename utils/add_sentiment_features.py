# Changes labels from category-specific labels to BIO-style labels.
# import sys
from spacy.lang.en import English


SENTIWORDS = '../data/sentiment/SentiWords_1.1.txt'
SENTIWORDNET = '../data/sentiment/SentiWordNet_3.0.0.txt'


def parse_sentiwordnet(lexicon_file):
    lex = dict()
    with open(lexicon_file, encoding='utf8') as f:
        for line in f:
            if line.startswith('#'):
                # comment
                continue
            fields = line.strip().split('\t')
            if len(fields) < 6:
                # last line
                continue
            # postag    id  score_pos   score_neg   word#sense word2#sense  def
            pos = float(fields[2])
            neg = float(fields[3])
            for word in fields[4].split():
                word = word.split('#')[0]
                try:
                    prev_pos, prev_neg, count = lex[word]
                    lex[word] = (prev_pos + pos, prev_neg + neg, count + 1)
                except KeyError:
                    lex[word] = (pos, neg, 1)

    for word in lex:
        pos, neg, count = lex[word]
        lex[word] = (pos / count, neg / count)

    return lex


def parse_sentiwords(lexicon_file):
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


def annotate(lex, infile, outfile, nlp, sentiwordnet):
    with open(infile, encoding='utf8') as f_in:
        with open(outfile, 'w', encoding='utf8') as f_out:
            first_line = True
            for line in f_in:

                # Comments + header
                if line.startswith('#'):
                    f_out.write(line)
                    continue
                if first_line:
                    f_out.write('# sentiment_lexicon=')
                    if sentiwordnet:
                        f_out.write('SentiWordNet')
                    else:
                        f_out.write('SentiWords')
                    f_out.write('\n')
                    first_line = False
                    labels = line.strip().split('\t')
                    try:
                        word_idx = labels.index('token')
                    except ValueError:
                        word_idx = 4
                    if sentiwordnet:
                        labels.append('positive')
                        labels.append('negative')
                    else:
                        labels.append('sentiment')
                    f_out.write('\t'.join(labels) + '\n')
                    continue

                line = line[:-1]  # Remove \n
                word = line.split('\t')[word_idx].lower()
                if sentiwordnet:
                    try:
                        value = lex[word]
                    except KeyError:
                        # Try looking up a lemmatized version
                        value = lex.get(nlp(word)[0].lemma_, (0.0, 0.0))
                    f_out.write(line + '\t' + str(value[0]) +
                                '\t' + str(value[1]) + '\n')
                else:
                    # SentiWords
                    try:
                        value = lex[word]
                    except KeyError:
                        # Try looking up a lemmatized version
                        value = lex.get(nlp(word)[0].lemma_, 0.0)
                    f_out.write(line + '\t' + str(value) + '\n')


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     sys.stderr.write('Usage:', sys.argv[0] + ' INFILE OUTFILE\n')
    #     sys.exit(1)

    # lex = parse_sentiwords(SENTIWORDS)
    # nlp = English()
    # annotate(lex, '../data/train-data-bio-improved.tsv',
    #          '../data/train-data-bio-improved-sentiment.tsv', nlp, False)
    # annotate(lex, '../data/dev-improved.tsv',
    #          '../data/dev-improved-sentiment.tsv', nlp, False)

    lex = parse_sentiwordnet(SENTIWORDNET)
    nlp = English()
    annotate(lex, '../data/train-improved.tsv',
             '../data/train-improved-sentiwordnet.tsv', nlp, True)
    annotate(lex, '../data/dev-improved.tsv',
             '../data/dev-improved-sentiwordnet.tsv', nlp, True)
    annotate(lex, '../data/test-improved.tsv',
             '../data/test-improved-sentiwordnet.tsv', nlp, True)
