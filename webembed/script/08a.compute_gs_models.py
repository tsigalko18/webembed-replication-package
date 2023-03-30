import csv
import glob
import json
import os
from pathlib import Path


def compute_gs_models():
    recap_data = []
    recap_headers = ['appname', 'number_of_bins', 'number_of_states', 'number_of_singleton_bins',
                     'number_of_bins_with_two_states', 'number_of_bins_with_more_than_two_states']

    # iterate over all the gold-standard files
    for filepath in glob.iglob('raw_data/*.json'):

        appname = Path(filepath).stem  # filename with no extension
        with open(filepath, 'r') as file:
            data = json.load(file)
            gold_standard = {}  # start with empty dict
            number_of_states = 0

            # iterate over each state in the gold-standard data
            for state_name, state_data in data['states'].items():
                number_of_states = number_of_states + 1
                if not state_data['bin'] in gold_standard.keys():  # it's the first time the current bin is seen
                    gold_standard[state_data['bin']] = []  # initialize to empty list
                gold_standard[state_data['bin']].append(state_name)  # and then append current state

            with open(f'./output/{appname}.json', 'w+') as output:
                json.dump(gold_standard, output)

            # compute some per-app statistics
            number_of_states = 0
            number_of_singleton_bins = 0
            number_of_bins_with_two_states = 0
            number_of_bins_with_more_than_two_states = 0

            for bin, states in gold_standard.items():
                number_of_states = number_of_states + len(states)
                if len(states) == 1:
                    number_of_singleton_bins = number_of_singleton_bins + 1
                elif len(states) == 2:
                    number_of_bins_with_two_states = number_of_bins_with_two_states + 1
                else:
                    number_of_bins_with_more_than_two_states = number_of_bins_with_more_than_two_states + 1

            recap_row = [
                appname,  # app name
                len(gold_standard.keys()),  # number of bins
                number_of_states,  # number of states
                number_of_singleton_bins,
                number_of_bins_with_two_states,
                number_of_bins_with_more_than_two_states
            ]

        recap_data.append(recap_row)

    with open('output/recap.csv', 'w+') as output:
        writer = csv.writer(output)
        writer.writerow(recap_headers)
        writer.writerows(recap_data)


if __name__ == '__main__':
    os.chdir("..")
    compute_gs_models()
