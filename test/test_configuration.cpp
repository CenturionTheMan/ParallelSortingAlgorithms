#include <iostream>

namespace testconf {
    const std::string VALID_CONFIG_FILE_CONTENT = 
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_bitonic=1\n"
        "measure_odd_even=1\n"
        "\n"
        "random_instance=50000 10\n"
        "random_instance=10000000 56\n"
        "random_instance=199 56\n"
        "\n"
        "predefined_instance=4 -1 88 2 9 4 105 1 34\n";


    const std::string CONFIG_FILE_WITH_NO_CPU_INFO =
        "measure_gpu=1\n"
        "measure_bitonic=1\n"
        "measure_odd_even=1\n";


    const std::string CONFIG_FILE_WITH_NO_GPU_INFO =
        "measure_bitonic=1\n"
        "measure_cpu=0\n"
        "measure_odd_even=1\n";


    const std::string CONFIG_FILE_WITH_NO_BITONIC_SORT_INFO =
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_odd_even=1\n";


    const std::string CONFIG_FILE_WITH_NO_ODD_EVEN_SORT_INFO =
        "measure_cpu=0\n"
        "measure_gpu=1\n"
        "measure_bitonic=1\n";
}