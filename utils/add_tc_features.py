


def add_question_marks(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tquestion\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4]
            if '?' in text:
                f.write('1\n')
            else:
                f.write('0\n')


add_question_marks('../data/tc-train.tsv')
add_question_marks('../data/tc-dev.tsv')
