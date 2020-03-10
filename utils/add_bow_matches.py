# Bag-of-words matches.

reductio_ad_hitlerum = ['hitler', 'nazi', 'fascis',  # -t,m
                        'stalin']


def match(text, words):
    for word in words:
        if word in text:
            return '1'
    return '0'


def annotate_file(words, in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\treductio\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            f.write(match(text, words))
            f.write('\n')


annotate_file(reductio_ad_hitlerum, '../data/tc-train.tsv')
annotate_file(reductio_ad_hitlerum, '../data/tc-dev.tsv')
annotate_file(reductio_ad_hitlerum, '../data/tc-test.tsv')
