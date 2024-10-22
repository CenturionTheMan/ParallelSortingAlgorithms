#include "test_configuration.h"

std::string testconf::readFileContents(const std::string &path) {
    std::ifstream file(path);
    if (!file.good())
        throw std::runtime_error("Test tried to access invalid file \"" + path + "\"");
    std::string file_contents;
    std::string line;
    while(!file.eof()){
        std::getline(file, line);
        file_contents += line + "\n";
    }
    file.close();
    return file_contents;
}