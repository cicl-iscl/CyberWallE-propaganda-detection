# Parses the argument lexicon by Somasundaran, Ruppenhofer & Wiebe.

import re
import sys


path = '../data/arglex/'
strategies = ['authority', 'doubt', 'emphasis', 'generalization', 'priority']
macros = ['modals', 'spoken', 'wordclasses', 'pronoun', 'intensifiers']
# macro -> list of expansions
expansions = dict()
# strategy -> list of regexes
regexes = dict()


def init(verbose=False):
    for macro in macros:
        with open(path + macro + '.tff') as f:
            for line in f:
                if line.startswith('#'):
                    # comment
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                fields = line.split('=')
                macro_word = fields[0]
                expansion_list = fields[1][1:-1]  # Strip away { and }
                # The lists use both ', ' and ',' as separators.
                expansion_list = expansion_list.replace(', ', ',')
                expansion_list = expansion_list.split(',')
                expansion_list = '(' + '|'.join(expansion_list) + ')'
                expansions[macro_word] = expansion_list

    if verbose:
        print('Macros and their expansions:')
        [print(m, expansions[m]) for m in expansions]
        print()

    for strategy in strategies:
        regexes[strategy] = []
        with open(path + strategy + '.tff') as f:
            for line in f:
                if line.startswith('#'):
                    # comment
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                for macro in expansions:
                    line = line.replace('(' + macro + ' )?',
                                        '(' + expansions[macro] + ' )?')
                    line = line.replace('(' + macro + ')', expansions[macro])
                line = line.replace('\\', '')
                regexes[strategy] += [line]

    if verbose:
        print('Regexes for rhetorical strategies:')
        [print(s, regexes[s]) for s in regexes]
        print()


def parse_demo_file(filename):
    with open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip().lower()
            line = line.replace('\\', '')  # Only for testing with data\arglex\patterntest.txt
            print(line)
            for strategy in regexes:
                for regex in regexes[strategy]:
                    for match in re.finditer(regex, line):
                        print(strategy.upper(), '--', match.group(),
                              '--', match.span(), '--', regex)
            print()


def find_rhetorical_strategies(token_list):
    sentence = ' '.join(token_list)
    token_indices = set()
    for strategy in regexes:
        for regex in regexes[strategy]:
            for match in re.finditer(regex, sentence):
                # print(strategy.upper(), '--', match.group(),
                #       '--', match.span(), '--', regex)
                start_idx = match.span()[0]
                end_idx = match.span()[1]
                # idx in the token list
                token_indices.add(sentence[:start_idx].count(' '))
                token_indices.add(sentence[:end_idx].count(' '))
    return token_indices


def parse_input_file(infile, outfile):
    with open(infile, encoding='utf8') as f_in:
        lines = f_in.readlines()
        lines.append('eof\teof\teof\teof\teof\teof\n')

    with open(outfile, 'w', encoding='utf8') as f_out:
        rows = []
        tokens = []
        prev_article = ''
        for line in lines:
            line = line[:-1]  # Remove \n
            fields = line.split('\t')
            article = fields[0]
            word = fields[4]
            if article != prev_article:
                indices = find_rhetorical_strategies(tokens)
                for i, row in enumerate(rows):
                    f_out.write(row)
                    f_out.write('\t')
                    if i in indices:
                        f_out.write('1')
                    else:
                        f_out.write('0')
                    f_out.write('\n')
                tokens = []
                rows = []
            tokens.append(word)
            rows.append(line)
            prev_article = article


# TODO read entire directories? or otherwise initialize the above only once
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     sys.stderr.write('Usage:', sys.argv[0] + 'FILENAME\n')
    #     sys.exit(1)

    init()
    # parse_demo_file(path + 'patterntest.txt')

    # toks = ['this', 'is', 'definitely', 'great', ';', 'it\'s', 'gonna', 'be', 'amazing']
    # indices = find_rhetorical_strategies(toks)
    # for i in indices:
    #     print(i, toks[i])

    parse_input_file('../data/train-data-bio-improved-sentiment.tsv',
                     '../data/train-data-bio-improved-sentiment-arguing.tsv')
    parse_input_file('../data/dev-improved-sentiment.tsv',
                     '../data/dev-improved-sentiment-arguing.tsv')
