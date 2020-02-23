FILE = '../data/train_dev_set.tsv'
DEV_WITHOUT_LABELS = '../data/dev-improved-sentiwordnet-arguingfullindiv-pos.tsv'
NEW_FILE = '../data/train_dev_set_fixed.tsv'


def check_sentence_numbers(infile, id_location=2):
    with open(infile, encoding='utf8') as f:
        prev_id = 0
        for line in f:
            try:
                sent_id = int(line.split('\t')[id_location])
            except ValueError:
                # Header
                continue
            if prev_id != sent_id and prev_id + 1 != sent_id:
                print(prev_id)
                print(line)
            prev_id = sent_id


# check_sentence_numbers(FILE)

# Insert dev sentences 1-21, update their IDs, add comments,
# remove the first column.

LAST_TRAIN_ID = 21501

with open(NEW_FILE, 'w', encoding='utf8') as f_out:
    with open(DEV_WITHOUT_LABELS, encoding='utf8') as f_dev:
        for line_dev in f_dev:
            if line_dev.strip().startswith('#'):
                # Comment
                f_out.write(line_dev)
            else:
                break
    with open(FILE, encoding='utf8') as f_in:
        train = True
        dev = False
        for line in f_in:
            cells = line.split('\t')
            try:
                sent_id = int(cells[2])
            except ValueError:
                # Header
                f_out.write('\t'.join(cells[1:]))
                continue
            if train and sent_id < LAST_TRAIN_ID:
                f_out.write('\t'.join(cells[1:]))
                continue
            if sent_id == LAST_TRAIN_ID:
                train = False
                f_out.write('\t'.join(cells[1:]))
                continue
            elif not dev:
                dev = True
                # Beginning of dev set: Insert sentences 1-21
                with open(DEV_WITHOUT_LABELS, encoding='utf8') as f_dev:
                    for line_dev in f_dev:
                        if line_dev.strip().startswith('#'):
                            # Comment
                            continue
                        cells_dev = line_dev.split('\t')
                        try:
                            sent_id_dev = int(cells_dev[1])
                        except ValueError:
                            # Header
                            continue
                        if sent_id_dev == 22:
                            break
                        f_out.write(cells_dev[0] + '\t')
                        f_out.write(str(sent_id_dev + LAST_TRAIN_ID) + '\t')
                        f_out.write('\t'.join(cells_dev[2:5]))
                        # The first 21 sentences don't contain propaganda spans
                        f_out.write('\tO\t')
                        f_out.write('\t'.join(cells_dev[5:]))
            f_out.write(cells[1] + '\t')
            f_out.write(str(sent_id + LAST_TRAIN_ID) + '\t')
            f_out.write('\t'.join(cells[3:]))
