import spacy
import re


REGEX = re.compile('\\bamerica\\b')


def search(nlp, text, features):
    results = {f: '0' for f in features}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in features:
            results[ent.label_] = '1'
    # Sometimes multi-word units containing a country name or nationality are
    # classified as 'ORG' instead of 'GPE'/'NORP'
    # -> Check individual tokens (out of context)
    if len(doc) > 1 and '0' in results.values():
        for tok in doc:
            for ent in nlp(tok.text).ents:
                if ent.label_ in features:
                    results[ent.label_] = '1'
    return results


def annotate_file_ner(nlp, features, in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip())
        for feature in features:
            f.write('\t' + feature)
        f.write('\n')
        for idx, line in enumerate(lines[1:]):
            f.write(line.strip())
            text = line.split('\t')[4]
            results = search(nlp, text, features)
            for feature in features:
                f.write('\t' + results[feature])
            f.write('\n')
            if idx % 250 == 0:
                print(idx, text, results)


def find_america(text):
    if 'america' not in text:
        return '0'
    for phrase in ['american people', 'americans', 'american citizen']:
        if phrase in text:
            return '1'
    if REGEX.search(text):
        return '1'
    return '0'


def annotate_file_america(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tamerica\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            f.write(find_america(text) + '\n')


def annotate_file_america_simple(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tamerica_simple\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            if 'america' in text:
                f.write('1\n')
            else:
                f.write('0\n')


nlp = spacy.load('en_core_web_sm')
features = ['NORP',  # Nationalities or religious or political groups
            'GPE']   # Geopolitical entities: Countries, cities, states
annotate_file_ner(nlp, features, '../data/tc-train.tsv')
annotate_file_ner(nlp, features, '../data/tc-dev.tsv')
annotate_file_ner(nlp, features, '../data/tc-test.tsv')

annotate_file_america('../data/tc-train.tsv')
annotate_file_america('../data/tc-dev.tsv')
annotate_file_america('../data/tc-test.tsv')

annotate_file_america_simple('../data/tc-train.tsv')
annotate_file_america_simple('../data/tc-dev.tsv')
annotate_file_america_simple('../data/tc-test.tsv')
