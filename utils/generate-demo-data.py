
# Generates a sample prediction file
# for testing the si_predictions_to_spans method in utils.py

with open('../data/train-data.tsv', encoding='utf8') as infile:
    with open('../data/si-sample-predictions.tsv', 'w', encoding='utf8') as outfile:
        lines = 476
        last_prediction = 'None'
        for line in infile:
            fields = line.strip().split('\t')
            outfile.write(fields[0])
            outfile.write('\t')
            outfile.write(fields[1])
            outfile.write('\t')
            outfile.write(fields[2])
            outfile.write('\t')
            outfile.write(fields[3])
            outfile.write('\t')
            if fields[4] == 'None':
                outfile.write('O')
            elif last_prediction == 'None':
                outfile.write('B')
            else:
                outfile.write('I')
            last_prediction = fields[4]
            outfile.write('\n')
            lines -= 1
            if lines == 0:
                break
