# Parses the argument lexicon by Somasundaran, Ruppenhofer & Wiebe.

import re
import sys


path = './data/arglex/'
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


def parse_file(filename):
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


# TODO read entire directories? or otherwise initialize the above only once
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     sys.stderr.write('Usage:', sys.argv[0] + 'FILENAME\n')
    #     sys.exit(1)

    init(verbose=True)
    parse_file(path + 'patterntest.txt')
    # parse_file(sys.argv[1])
