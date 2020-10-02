# An Empirical Study of Example Forgetting during Deep Neural Network Learning

This repository contains code for the paper [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159).

Bibtex: 
```
@inproceedings{Forgetting,
    title={An Empirical Study of Example Forgetting during Deep Neural Network Learning},
    author={Toneva, Mariya and Sordoni, Alessandro and Combes, Remi Tachet des and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J},
    booktitle={ICLR},
    year={2019}
}
```

## Code Requirements
The environment requirements for the code in this repository are provided in requirements.txt. They can be installed in bulk with the following command: 
```
pip install -r requirements.txt
```


## Computing Forgetting Counts

<p align="center">
  <img width="800" src="https://github.com/mtoneva/forgetting_temp/blob/master/figures/Fig1_nocaption.png">
</p>

Our approach consists of two main steps:
1. Train on full training set to collect statistics (loss, accuracy, misclassification margin) after each presentation of every example.
2. Given these presentation statistics, compute number of forgetting events per example and sort examples by forgetting counts. A forgetting event is defined as a transition in the training accuracy of an example from 1 to 0 on two consecutive presentations. Note that a missclassification is not necessarily a forgetting event.

We present results on MNIST and CIFAR. Below we provide an overview of the supplied code and commands to reproduce our results for both steps.

##### MNIST and permuted MNIST:

```
python run_mnist.py 
    --dataset [mnist/permuted_mnist]
    --no_dropout 
    --output_dir [mnist/permuted_mnist]_results
    --seed s
```
, where s ranges from to 1 to 5. The default setting was used for all other flags. Use options `--dataset mnist` and `--output_dir mnist_results` to run on MNIST, and options `--dataset permuted_mnist` and `--output_dir permuted_mnist_results` for permuted MNIST. Each training run with a different seed saves a file that contains the presentation statistics (loss, accuracy, misclassification margin) from that run in the specified `--output_dir`. The names of the saved files contain the arguments (and argument values) that were used to generate them.

```
python order_examples_by_forgetting.py 
    --output_dir [mnist/permuted_mnist]_results 
    --output_name [mnist/permuted_mnist]_sorted 
    --input_dir [mnist/permuted_mnist]_results 
    --input_fname_args 
            dataset [mnist/permuted_mnist] 
            no_dropout True 
            sorting_file none 
            remove_n 0 
            keep_lowest_n 0
```
This finds all output files produced by `run_mnist.py` that are in `--input_dir` and match the arguments and values specified by `--input_fname_args`. Note that `--seed` is not specified above, which enables us to match the output files of all 5 training runs. Using all matched files, `order_examples_by_forgetting.py` calculates the total number of times an example is forgotten across all epochs of all training runs (i.e. the example's forgetting counts). Then, all examples are sorted by their forgetting counts in ascending order (i.e. the examples that are unforgettable, or forgotten 0 times, come first), and the sorted examples and their respective total forgetting counts are saved in a dictionary with the specified name in `--output_fname`. This script also outputs the number of unforgettable examples across all completed training runs. 

##### CIFAR-10 and CIFAR-100:
```
python run_cifar.py 
    --dataset [cifar10/cifar100] 
    --data_augmentation 
    --output_dir [cifar10/cifar100]_results
    --seed s
```
, where s ranges from to 1 to 5. The default setting was used for all other flags. This script has a similar functionality to `run_mnist.py`.

```
python order_examples_by_forgetting.py 
    --output_dir [cifar10/cifar100]_results 
    --output_name [cifar10/cifar100]_sorted 
    --input_dir [cifar10/cifar100]_results 
    --input_fname_args 
            dataset [cifar10/cifar100]
            data_augmentation True 
            cutout False 
            sorting_file none 
            remove_n 0 
            keep_lowest_n 0 
            remove_subsample 0 
            noise_percent_labels 0 
            noise_percent_pixels 0 
            noise_std_pixels 0
```


## Experiments

#### Removing examples from training set

<p align="center">
  <img width="800" src="https://github.com/mtoneva/forgetting_temp/blob/master/figures/Fig5_nocaption.png">
</p>

The removal experiments specify a number of examples to be completely removed from the sorted training set. We achieve this by providing three extra flags to `run_mnist.py` and `run_cifar.py`: `--sorting_file`, which is the name of the file output by `order_examples_by_forgetting.py` that specifies the sorting of the examples based on forgetting counts, `--remove_n`, which specifies the number of examples to remove, and `--keep_lowest_n`, which specifies where in the list of sorted training examples the removal should begin. We found that near state-of-the-art generalization performance can be maintained even when all unforgettable examples (i.e. examples with 0 forgetting events) are removed. 

For Figure 5 Left results:
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --cutout 
    --sorting_file cifar10_sorted 
    --input_dir cifar10_results 
    --output_dir cifar10_results 
    --seed s 
    --remove_n r 
    --keep_lowest_n k
```
, where s is in `range(1,6)`, r is in `range(0,50000,1000)`, and k is 0 (for selected) and -1 (for random).

For Figure 5 Right results:
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --cutout 
    --sorting_file cifar10_sorted 
    --input_dir cifar10_results 
    --output_dir cifar10_results 
    --seed s 
    --remove_n r 
    --keep_lowest_n k
```
, where s is in `range(1,6)`, r is 5000, and k is in `range(0,50000,1000)` (for selected) and -1 (for random).

Cutout implementation from [Cutout repository](https://github.com/uoguelph-mlrg/Cutout).

For Figure 6 results:
```
python run_mnist.py 
    --dataset [mnist/permuted_mnist] 
    --sorting_file [mnist/permuted_mnist]_sorted 
    --input_dir [mnist/permuted_mnist]_results 
    --output_dir [mnist/permuted_mnist]_results  
    --seed s 
    --remove_n r 
    --keep_lowest_n k
```
, where s is in `range(1,6)`, r is in `range(0,60000,1000)`, and k is 0 (for selected) and -1 (for random). The results for CIFAR-10 in this figure are the same results as those from Figure 5 Left. 

#### Adding label noise during training

We also investigate how adding noise to the example labels affects forgetting. We introduce label noise by assigning random labels to a specified percentage of the training set.

For Figure 3 results:
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --output_dir cifar10_results 
    --noise_percent_labels 20
```

#### Adding pixel noise during training

In the supplementary, we further investigate how introducing additive Gaussian noise to the example pixels affects forgetting.

For Supplementary Figure 11 results:
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --output_dir cifar10_results 
    --noise_percent_pixels 100 
    --noise_std_pixels n
```
, where n is in `[0.5,1,2,10]`.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
