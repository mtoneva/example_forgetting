import argparse
import numpy as np
import os
import pickle


# Calculates forgetting statistics per example
#
# diag_stats: dictionary created during training containing 
#             loss, accuracy, and missclassification margin 
#             per example presentation
# npresentations: number of training epochs
#
# Returns 4 dictionaries with statistics per example
#
def compute_forgetting_statistics(diag_stats, npresentations):

    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


# Sorts examples by number of forgetting counts during training, in ascending order
# If an example was never learned, it is assigned the maximum number of forgetting counts
# If multiple training runs used, sort examples by the sum of their forgetting counts over all runs
#
# unlearned_per_presentation_all: list of dictionaries, one per training run
# first_learned_all: list of dictionaries, one per training run
# npresentations: number of training epochs
#
# Returns 2 numpy arrays containing the sorted example ids and corresponding forgetting counts
#
def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):

    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats)


# Checks whether a given file name matches a list of specified arguments
#
# fname: string containing file name
# args_list: list of strings containing argument names and values, i.e. [arg1, val1, arg2, val2,..]
#
# Returns 1 if filename matches the filter specified by the argument list, 0 otherwise
#
def check_filename(fname, args_list):

    # If no arguments are specified to filter by, pass filename
    if args_list is None:
        return 1

    for arg_ind in np.arange(0, len(args_list), 2):
        arg = args_list[arg_ind]
        arg_value = args_list[arg_ind + 1]

        # Check if filename matches the current arg and arg value
        if arg + '_' + arg_value + '__' not in fname:
            print('skipping file: ' + fname)
            return 0

    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument(
        '--input_fname_args',
        nargs='+',
        help=
        'arguments and argument values to select input filenames, i.e. arg1 val1 arg2 val2'
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--output_name',
        type=str,
        required=True)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    print(args)

    # Initialize lists to collect forgetting stastics per example across multiple training runs
    unlearned_per_presentation_all, first_learned_all = [], []

    for d, _, fs in os.walk(args.input_dir):
        for f in fs:

            # Find the files that match input_fname_args and compute forgetting statistics
            if f.endswith('stats_dict.pkl') and check_filename(
                    f, args.input_fname_args):
                print('including file: ' + f)

                # Load the dictionary compiled during training run
                with open(os.path.join(d, f), 'rb') as fin:
                    loaded = pickle.load(fin)

                # Compute the forgetting statistics per example for training run
                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
                    loaded, args.epochs)

                unlearned_per_presentation_all.append(
                    unlearned_per_presentation)
                first_learned_all.append(first_learned)

    if len(unlearned_per_presentation_all) == 0:
        print('No input files found in {} that match {}'.format(
            args.input_dir, args.input_fname_args))
    else:

        # Sort examples by forgetting counts in ascending order, over one or more training runs
        ordered_examples, ordered_values = sort_examples_by_forgetting(
            unlearned_per_presentation_all, first_learned_all, args.epochs)

        # Save sorted output
        if args.output_name.endswith('.pkl'):
            with open(os.path.join(args.output_dir, args.output_name),
                      'wb') as fout:
                pickle.dump({
                    'indices': ordered_examples,
                    'forgetting counts': ordered_values
                }, fout)
        else:
            with open(
                    os.path.join(args.output_dir, args.output_name + '.pkl'),
                    'wb') as fout:
                pickle.dump({
                    'indices': ordered_examples,
                    'forgetting counts': ordered_values
                }, fout)
