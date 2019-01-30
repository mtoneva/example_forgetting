import pdb
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from Cutout.util.misc import CSVLogger
from Cutout.util.cutout import Cutout

from Cutout.model.resnet import ResNet18
from Cutout.model.wide_resnet import WideResNet


# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


# Introduce Gaussian noise to noise_percentage of image pixels
def noisy(image, noise_percentage, noise_std):
    row, col, ch = image.shape
    num_corrupt = int(np.floor(noise_percentage * row * col / 100))

    # Randomly choose pixels to add noise to
    xy_coords = np.random.choice(row * col, num_corrupt, replace=False)
    chan_coords = np.random.choice(ch, num_corrupt, replace=True)
    xy_coords = np.unravel_index(xy_coords, (row, col))

    out = np.copy(image)

    mean = 120

    # Add randomly generated Gaussian noise to pixels
    for coord in range(num_corrupt):
        noise = np.random.normal(mean, noise_std, 1)
        out[xy_coords[0][coord], xy_coords[1][coord],
            chan_coords[coord]] += noise

    return out


# Train model for one epoch
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def train(args, model, device, trainset, model_optimizer, epoch,
          example_stats):
    train_loss = 0.
    correct = 0.
    total = 0.

    model.train()

    # Get permutation to shuffle trainset
    trainset_permutation_inds = npr.permutation(
        np.arange(len(trainset.train_labels)))

    print('\n=> Training Epoch #%d' % (epoch))

    batch_size = args.batch_size
    for batch_idx, batch_start_ind in enumerate(
            range(0, len(trainset.train_labels), batch_size)):

        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]

        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(
            np.array(trainset.train_labels)[batch_inds].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        model_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        # Update statistics and loss
        acc = predicted == targets
        for j, index in enumerate(batch_inds):

            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = train_indx[index]

            # Compute missclassification margin
            output_correct_class = outputs.data[j, targets[j].item()]
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats.get(index_in_original_dataset,
                                            [[], [], []])
            index_stats[0].append(loss[j].item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats[index_in_original_dataset] = index_stats

        # Update loss, backward propagate, update optimizer
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        model_optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_idx + 1,
             (len(trainset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()

        # Add training accuracy to dict
        index_stats = example_stats.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['train'] = index_stats


# Evaluate model predictions on heldout test data
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def test(epoch, model, device, example_stats):
    global best_acc
    test_loss = 0.
    correct = 0.
    total = 0.
    test_batch_size = 32

    model.eval()

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(test_dataset.test_labels), test_batch_size)):

        # Get batch inputs and targets
        transformed_testset = []
        for ind in range(
                batch_start_ind,
                min(
                    len(test_dataset.test_labels),
                    batch_start_ind + test_batch_size)):
            transformed_testset.append(test_dataset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_testset)
        targets = torch.LongTensor(
            np.array(
                test_dataset.test_labels)[batch_start_ind:batch_start_ind +
                                          test_batch_size].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Add test accuracy to dict
    acc = 100. * correct.item() / total
    index_stats = example_stats.get('test', [[], []])
    index_stats[1].append(100. * correct.item() / float(total))
    example_stats['test'] = index_stats
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))

    # Save checkpoint when best model
    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        save_point = os.path.join(args.output_dir, 'checkpoint', args.dataset)
        os.makedirs(save_point, exist_ok=True)
        torch.save(state, os.path.join(save_point, save_fname + '.t7'))
        best_acc = acc


model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', default='cifar10', choices=dataset_options)
parser.add_argument('--model', default='resnet18', choices=model_options)
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='input batch size for training (default: 128)')
parser.add_argument(
    '--epochs',
    type=int,
    default=200,
    help='number of epochs to train (default: 200)')
parser.add_argument(
    '--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument(
    '--data_augmentation',
    action='store_true',
    default=False,
    help='augment data by flipping and cropping')
parser.add_argument(
    '--cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument(
    '--n_holes',
    type=int,
    default=1,
    help='number of holes to cut out from image')
parser.add_argument(
    '--length', type=int, default=16, help='length of the holes')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='enables CUDA training')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--sorting_file',
    default="none",
    help=
    'name of a file containing order of examples sorted by forgetting (default: "none", i.e. not sorted)'
)
parser.add_argument(
    '--remove_n',
    type=int,
    default=0,
    help='number of sorted examples to remove from training')
parser.add_argument(
    '--keep_lowest_n',
    type=int,
    default=0,
    help=
    'number of sorted examples to keep that have the lowest score, equivalent to start index of removal, if a negative number given, remove random draw of examples'
)
parser.add_argument(
    '--remove_subsample',
    type=int,
    default=0,
    help='number of examples to remove from the keep-lowest-n examples')
parser.add_argument(
    '--noise_percent_labels',
    type=int,
    default=0,
    help='percent of labels to randomly flip to a different label')
parser.add_argument(
    '--noise_percent_pixels',
    type=int,
    default=0,
    help='percent of pixels to randomly introduce Gaussian noise to')
parser.add_argument(
    '--noise_std_pixels',
    type=float,
    default=0,
    help='standard deviation of Gaussian pixel noise')
parser.add_argument(
    '--optimizer',
    default="sgd",
    help='optimizer to use, default is sgd. Can also use adam')
parser.add_argument(
    '--input_dir',
    default='cifar10_results/',
    help='directory where to read sorting file from')
parser.add_argument(
    '--output_dir', required=True, help='directory where to save results')

# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'dataset', 'data_augmentation', 'cutout', 'seed', 'sorting_file',
    'remove_n', 'keep_lowest_n', 'remove_subsample', 'noise_percent_labels',
    'noise_percent_pixels', 'noise_std_pixels'
]

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
save_fname = '__'.join(
    '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set appropriate devices
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = args.cuda
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True  # Should make training go faster for large models

# Set random seed for initialization
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

# Image Preprocessing
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

# Setup train transforms
train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(
        Cutout(n_holes=args.n_holes, length=args.length))

# Setup test transforms
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

os.makedirs(args.output_dir, exist_ok=True)

# Load the appropriate train and test datasets
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(
        root='/tmp/data/',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR10(
        root='/tmp/data/',
        train=False,
        transform=test_transform,
        download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=False,
        transform=test_transform,
        download=True)

# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(train_dataset.train_labels)))
else:
    try:
        with open(
                os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']
    except IOError:
        with open(os.path.join(args.input_dir, args.sorting_file),
                  'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']

    # Get the indices to remove from training
    elements_to_remove = np.array(
        ordered_indx)[args.keep_lowest_n:args.keep_lowest_n + args.remove_n]

    # Remove the corresponding elements
    train_indx = np.setdiff1d(
        range(len(train_dataset.train_labels)), elements_to_remove)

if args.keep_lowest_n < 0:
    # Remove remove_n number of examples from the train set at random
    train_indx = npr.permutation(np.arange(len(
        train_dataset.train_labels)))[:len(train_dataset.train_labels) -
                                      args.remove_n]

elif args.remove_subsample:
    # Remove remove_sample number of examples at random from the first keep_lowest_n examples
    # Useful when the first keep_lowest_n examples have equal forgetting counts
    lowest_n = np.array(ordered_indx)[0:args.keep_lowest_n]
    train_indx = lowest_n[npr.permutation(np.arange(
        args.keep_lowest_n))[:args.keep_lowest_n - args.remove_subsample]]
    train_indx = np.hstack((train_indx,
                            np.array(ordered_indx)[args.keep_lowest_n:]))

# Reassign train data and labels
train_dataset.train_data = train_dataset.train_data[train_indx, :, :, :]
train_dataset.train_labels = np.array(
    train_dataset.train_labels)[train_indx].tolist()

# Introduce noise to images if specified
if args.noise_percent_pixels:
    for ind in range(len(train_indx)):
        image = train_dataset.train_data[ind, :, :, :]
        noisy_image = noisy(image, args.noise_percent_pixels, args.noise_std_pixels)
        train_dataset.train_data[ind, :, :, :] = noisy_image

# Introduce noise to labels if specified
if args.noise_percent_labels:
    fname = os.path.join(args.output_dir, save_fname)

    with open(fname + "_changed_labels.txt", "w") as f:

        # Compute number of labels to change
        nlabels = len(train_dataset.train_labels)
        nlabels_to_change = int(args.noise_percent_labels * nlabels / 100)
        nclasses = len(np.unique(train_dataset.train_labels))
        print('flipping ' + str(nlabels_to_change) + ' labels')

        # Randomly choose which labels to change, get indices
        labels_inds_to_change = npr.choice(
            np.arange(nlabels), nlabels_to_change, replace=False)

        # Flip each of the randomly chosen labels
        for l, label_ind_to_change in enumerate(labels_inds_to_change):

            # Possible choices for new label
            label_choices = np.arange(nclasses)

            # Get true label to remove it from the choices
            true_label = train_dataset.train_labels[label_ind_to_change]

            # Remove true label from choices
            label_choices = np.delete(
                label_choices,
                true_label)  # the label is the same as the index of the label

            # Get new label and relabel the example with it
            noisy_label = npr.choice(label_choices, 1)
            train_dataset.train_labels[label_ind_to_change] = noisy_label[0]

            # Write the example index from the original example order, the old, and the new label
            f.write(
                str(train_indx[label_ind_to_change]) + ' ' + str(true_label) +
                ' ' + str(noisy_label[0]) + '\n')

print('Training on ' + str(len(train_dataset.train_labels)) + ' examples')

# Setup model
if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        model = WideResNet(
            depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.4)
    else:
        model = WideResNet(
            depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
else:
    print(
        'Specified model not recognized. Options are: resnet18 and wideresnet')

# Setup loss
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion.__init__(reduce=False)

# Setup optimizer
if args.optimizer == 'adam':
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif args.optimizer == 'sgd':
    model_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4)
    scheduler = MultiStepLR(
        model_optimizer, milestones=[60, 120, 160], gamma=0.2)
else:
    print('Specified optimizer not recognized. Options are: adam and sgd')

# Initialize dictionary to save statistics for every example presentation
example_stats = {}

best_acc = 0
elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, train_dataset, model_optimizer, epoch,
          example_stats)
    test(epoch, model, device, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Update optimizer step
    if args.optimizer == 'sgd':
        scheduler.step(epoch)

    # Save the stats dictionary
    fname = os.path.join(args.output_dir, save_fname)
    with open(fname + "__stats_dict.pkl", "wb") as f:
        pickle.dump(example_stats, f)

    # Log the best train and test accuracy so far
    with open(fname + "__best_acc.txt", "w") as f:
        f.write('train test \n')
        f.write(str(max(example_stats['train'][1])))
        f.write(' ')
        f.write(str(max(example_stats['test'][1])))
