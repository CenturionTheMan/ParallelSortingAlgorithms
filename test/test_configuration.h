#ifndef TEST_CONFIGURATION_H
#define TEST_CONFIGURATION_H

#include <fstream>
#include <iostream>
#include <string>


namespace testconf {
    std::string readFileContents(const std::string& path);
}


namespace testdata {
    const std::string VALID_CONFIG_FILE_CONTENT = 
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_bitonic=1\n"
        "measure_odd_even=1\n"
        "verify=0\n"
        "\n"
        "random_instance=50000 10\n"
        "random_instance=10000000 56\n"
        "random_instance=199 56\n"
        "\n"
        "predefined_instance=4 -1 88 2 9 4 105 1 34\n";


    const std::string CONFIG_FILE_WITH_NO_CPU_INFO =
        "measure_gpu=1\n"
        "measure_bitonic=1\n"
        "measure_odd_even=1\n"
        "verify=0\n";


    const std::string CONFIG_FILE_WITH_NO_GPU_INFO =
        "measure_bitonic=1\n"
        "measure_cpu=0\n"
        "measure_odd_even=1\n"
        "verify=0\n";


    const std::string CONFIG_FILE_WITH_NO_BITONIC_SORT_INFO =
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_odd_even=1\n"
        "verify=0\n";


    const std::string CONFIG_FILE_WITH_NO_ODD_EVEN_SORT_INFO =
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_bitonic=1\n";

    const std::string CONFIG_FILE_WITH_NO_VERIFY_INFO =
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_bitonic=1\n"
        "measure_odd_even=1\n";

    const std::string PRINT_FOR_VALID_CONFIGURATION = 
        ">>> CONFIGURATION LOADED\n"
        "\n"
        "CPU measurement                 OFF\n"
        "GPU measurement                 ON\n"
        "Bitonic Sort measurement        ON\n"
        "Odd-Even Sort measurement       ON\n"
        "Verify                          OFF\n"
        "\n"
        "Defined instances               4\n"
        "\n";
    

    const std::string PRINT_FOR_CPU_INVALID_SOLUTION =
        ">>> BENCHMARK TERMINATED!\n"
        ">>> The CPU algorithm_1 has given an invalid solution for instance size 10 in repetition 5.\n"
        ">>> Please check the \"error.log\" file.\n";


    const std::string ERROR_FILE_FOR_CPU_INVALID_SOLUTION =
        ">>> The CPU algorithm_1 has given an invalid solution for instance size 10 in repetition 5.\n"
        "[Instance]: 8 9 3 2 10 5 6 7 4 1\n"
        "[Solution]: 8 9 3 2 10 5 6 7 4 1\n";
    
    
    const std::string PRINT_FOR_GPU_INVALID_SOLUTION =
        ">>> BENCHMARK TERMINATED!\n"
        ">>> The GPU algorithm_1 has given an invalid solution for instance size 10 in repetition 7.\n"
        ">>> Please check the \"error.log\" file.\n";


    const std::string ERROR_FILE_FOR_GPU_INVALID_SOLUTION =
        ">>> The GPU algorithm_1 has given an invalid solution for instance size 10 in repetition 7.\n"
        "[Instance]: 8 9 3 2 10 5 6 7 4 1\n"
        "[Solution]: 8 9 3 2 10 5 6 7 4 1\n";
    

    const std::string PRINT_FOR_BOTH_INVALID_SOLUTIONS =
        ">>> BENCHMARK TERMINATED!\n"
        ">>> Both implementations of algorithm_1 has given an invalid solution for instance size 10 in repetition 2137.\n"
        ">>> Please check the \"error.log\" file.\n";


    const std::string ERROR_FILE_FOR_BOTH_INVALID_SOLUTIONS =
        ">>> Both implementations of algorithm_1 has given an invalid solution for instance size 10 in repetition 2137.\n"
        "[Instance]: 8 9 3 2 10 5 6 7 4 1\n"
        "[Solution]: 8 9 3 2 10 5 6 7 4 1\n";
    

    const std::string RESULT_FILE_HEADER =
        "instance size;mean bitonic sort (CPU);mean bitonic sort (GPU);mean odd-even sort (CPU);mean odd-event sort (GPU)\n";

    
    const std::string RESULTS_CSV_ALL_ENABLED =
        RESULT_FILE_HEADER + 
        "10;1.500000;2.400000;3.600000;4.500000\n"
        "\n";
    
    const std::string RESULTS_CSV_SOME_DISABLED =
        RESULT_FILE_HEADER +
        "10;1.500000;;3.600000;\n"
        "\n";

    
    const std::string TABLE_HEADER =
        ">>> STARTING BENCHMARK...\n"
        "\n"
        "#===============#=========================#=========================#==========================#==========================#\n"
        "| Instance size |       CPU Bitonic       |       GPU Bitonic       |       CPU Odd-Even       |       GPU Odd-Even       |\n"
        "#===============#=========================#=========================#==========================#==========================#\n";
    

    const std::string TABLE_WITH_ALL_RESULTS =
        TABLE_HEADER +
        "|            10 |   1.50e+00 (1.50e+00) s |   2.40e+00 (2.40e+00) s |    3.60e+00 (3.60e+00) s |    4.50e+00 (4.50e+00) s |\n"
        "#===============#=========================#=========================#==========================#==========================#\n";

    
    const std::string TABLE_WITH_PARTIAL_RESULTS =
        TABLE_HEADER +
        "|            10 |   1.50e+00 (1.50e+00) s |                         |    3.60e+00 (3.60e+00) s |                          |\n"
        "#===============#=========================#=========================#==========================#==========================#\n";
}

#endif