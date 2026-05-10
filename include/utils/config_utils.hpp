#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <rfl/DefaultIfMissing.hpp>
#include <rfl/toml.hpp>

inline std::vector<std::string> loadClassNamesFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open class names file: " + filename);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(file, line))
        if (!line.empty())
            names.push_back(line);
    return names;
}

template <typename T>
T loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open config file: " + filename);
    auto result = rfl::toml::read<T, rfl::DefaultIfMissing>(file);
    if (!result)
        throw std::runtime_error(result.error().what());
    return *result;
}
