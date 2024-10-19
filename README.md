# ParallelSortingAlgorithms

## Benchmarking tool

This project includes a source code for CLI benchmarking tool that allows you to measure time complexities of *Bitonic* and *Odd-Even* array sorting algorithms implemented for both CPU (plain *C++*) and GPU (*C++* & *CUDA*).

### Functionalities

#### Time measurement

Tool allows to measure time complexities for CPU and GPU implementations of *Bitonic* and *Odd-Even* array sorting algorithms. The measurement would be performed for different sizes of randomly-generated or predefined sorting problem instances. Each instance size would be measured multiple times in order to calculate an average result.

When measurement is done the average results for each instance size would be printed in following tabular format:

```
>>> STARTING BENCHMARK...

#=============================#=============#==============#==============#
| Instance size | CPU Bitonic | GPU Bitonic | CPU Odd-Even | GPU Odd-Even |
#=============================#=============#==============#==============#
|       1000000 | 1.23e+456 s | 1.23e+456 s |  1.23e+456 s |  1.23e+456 s |
|      10000000 | 1.23e+456 s | 1.23e+456 s |  1.23e+456 s |  1.23e+456 s |
|     100000000 | 1.23e+456 s | 1.23e+456 s |  1.23e+456 s |  1.23e+456 s |
|    1000000000 | 1.23e+456 s | 1.23e+456 s |  1.23e+456 s |  1.23e+456 s |
#=============================#=============#==============#==============#

>>> BENCHMARK COMPLETE!
```

#### Saving results

System saves measurement results for each repetition in an output file named `result.csv` located in tool's parent directory. Each line is associated with one repetition and contains following semicolon-separated (`;`) fields:

1. Instance size (integer)
2. Execution time for *Bitonic Sort* implemented on CPU (real)
3. Execution time for *Bitonic Sort* implemented on GPU (real)
3. Execution time for *Odd-Even Sort* implemented on CPU (real)
3. Execution time for *Odd-Even Sort* implemented on GPU (real)

What's more the header would be also saved to the first line of `result.csv`.

#### Customizable configuration

Tool would allow user to specify custom configuration in the `configuration.ini` file. This file must be located in tool's parent directory. Each line of configuration file contains a key-value pairs with format `key=value`. All of the required keys are listed below:

- `measure_cpu`: Boolean value that defines if measurement for CPU implementations would be performed.
- `measure_gpu`: Boolean value that defines if measurement for GPU implementations would be performed.
- `measure_bitonic`: Boolean value that defines if measurement for *Bitonic Sort* would be performed.
- `measure_odd_even`: Boolean value that defines if measurement for *Odd-Even Sort* would be performed.

Other lines in this section may contain data for instances that should be measured. Each line contains a key-value pair where key is an `random_instance` or `predefined_instance` keyword. For `random_instance` key the value is a space-separated pair of instance size and number of measurement repetitions for it. For `predefined_instance` key the value is a number of repetitions for instance followed by the integers that are a part of instance itself (all space-separated).

The configuration file may also contain empty lines.

Example of a valid configuration file is shown below:

```ini
measure_cpu=0
measure_gpu=1
measure_bitonic=1

random_instance=50000 10
random_instance=10000000 56
random_instance=199 56

predefined_instance=4 -1 88 2 9 4 105 1 34
```

#### Verification of solutions

For each repetition solutions from all of the implementations are verified. If implementation sorted instance properly then nothing special happens. Otherwise the tool would be terminated and information about an error would be printed to console (example below).

```
>>> STARTING BENCHMARK...

#=============================#=============#==============#==============#
| Instance size | CPU Bitonic | GPU Bitonic | CPU Odd-Even | GPU Odd-Even |
#=============================#=============#==============#==============#
|       1000000 | 1.23e+456 s | 1.23e+456 s |  1.23e+456 s |  1.23e+456 s |

>>> BENCHMARK TERMINATED!
>>> The GPU Odd-Even has given an invalid solution for instance size 100000 in repetition 4.
>>> Please check the "error.log" file.
```

Details about an error would be saved to the `error.log` file that would be located in the tool's parent directory. This details are:

- Error message from console.
- List of space-separated integers that are part of problematic instance.
- Invalidly sorted instance in a form of space-separated integers that are a part of problematic instance.

Example of `error.log` content is shown below:

```log
>>> The GPU Odd-Even has given an invalid solution for instance size 10  in repetition 4.
[Instance]: 0 5 8 1 3 5 8
[Solution]: 0 1 3 8 5 8 5
```

### How to compile and use benchmarking tool

In order to use benchmarking tool you need to compile the project. We recommend using a `gcc` compiler just as it's shown below.

```bash
gcc 
```

### Sorting function convention

## Resources

- [Project's board on Trello](https://trello.com/b/PZKg8jf4/gpuwt0910zrownoleglonealgorytmysortowania)