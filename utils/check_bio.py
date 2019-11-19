with open('../data/train-data-bio-baseline.tsv', encoding='utf8') as f:
    prev_label = ''
    prev_line = ''
    for line in f:
        line = line.strip()
        label = line[-1]
        if label == 'I' and prev_label == 'O':
            print(prev_line)
            print(line)
            print()
        prev_label = label
        prev_line = line
