#include <fstream>

#include <gtest/gtest.h>

#include "test_configuration.h"
#include "config.h"


TEST(loadConfiguration, when_file_exist_then_load_configuration) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::VALID_CONFIG_FILE_CONTENT;
    configuration_file.close();

    config::configuration_t expected_config = config::configuration_t(true, false, true, true, false);
    expected_config.loaded_instances.emplace(data::instance_t(std::list<int>(50000), 10));
    expected_config.loaded_instances.emplace(data::instance_t(std::list<int>(10000000), 56));
    expected_config.loaded_instances.emplace(data::instance_t(std::list<int>(199), 56));
    expected_config.loaded_instances.emplace(data::instance_t({-1, 88, 2, 9, 4, 105, 1, 34}, 4));

    config::configuration_t actual_config = config::loadConfiguration();

    ASSERT_EQ(actual_config.measure_bitonic, expected_config.measure_bitonic);
    ASSERT_EQ(actual_config.measure_cpu, expected_config.measure_cpu);
    ASSERT_EQ(actual_config.measure_gpu, expected_config.measure_gpu);
    ASSERT_EQ(actual_config.measure_odd_even, expected_config.measure_odd_even);
    ASSERT_EQ(actual_config.loaded_instances.size(), expected_config.loaded_instances.size());

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_not_exist_then_throw_runtime_error) {
    EXPECT_THROW(config::loadConfiguration(), std::runtime_error);
}


TEST(loadConfiguration, when_file_not_exist_then_print_error_message) {
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::runtime_error& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(), 
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file is invalid or doesn't exist!\n"
    );
}


TEST(loadConfiguration, when_file_has_no_cpu_info_then_throw_runtime_error) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_CPU_INFO;
    configuration_file.close();

    EXPECT_THROW(config::loadConfiguration(), std::runtime_error);

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_cpu_info_then_print_error_message) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_CPU_INFO;
    configuration_file.close();
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::exception& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(),
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file misses \"measure_cpu\"!\n"
    );
    

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_gpu_info_then_throw_runtime_error) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_GPU_INFO;
    configuration_file.close();

    EXPECT_THROW(config::loadConfiguration(), std::runtime_error);

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_gpu_info_then_print_error_message) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_GPU_INFO;
    configuration_file.close();
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::exception& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(),
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file misses \"measure_gpu\"!\n"
    );
    

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_bitonic_sort_info_then_throw_runtime_error) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_BITONIC_SORT_INFO;
    configuration_file.close();

    EXPECT_THROW(config::loadConfiguration(), std::runtime_error);

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_bitonic_sort_info_then_print_error_message) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_BITONIC_SORT_INFO;
    configuration_file.close();
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::exception& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(),
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file misses \"measure_bitonic\"!\n"
    );

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_odd_even_sort_info_then_throw_runtime_error) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_ODD_EVEN_SORT_INFO;
    configuration_file.close();

    EXPECT_THROW(config::loadConfiguration(), std::runtime_error);

    std::remove("configuration.ini");
}


TEST(loadConfiguration, when_file_has_no_odd_even_sort_info_then_print_error_message) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_ODD_EVEN_SORT_INFO;
    configuration_file.close();
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::exception& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(),
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file misses \"measure_odd_even\"!\n"
    );

    std::remove("configuration.ini");
}

TEST(loadConfiguration, when_file_has_no_verify_info_then_print_error_message) {
    std::fstream configuration_file("configuration.ini", std::ios::out);
    configuration_file << testdata::CONFIG_FILE_WITH_NO_VERIFY_INFO;
    configuration_file.close();
    testing::internal::CaptureStdout();

    try{
        config::loadConfiguration();
    } catch(const std::exception& e){}

    EXPECT_EQ(
        testing::internal::GetCapturedStdout(),
        ">>> BENCHMARK TERMINATED!\n"
        ">>> ERROR: Configuration file misses \"verify\"!\n"
    );

    std::remove("configuration.ini");
}