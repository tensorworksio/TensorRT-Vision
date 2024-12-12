#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

struct Detection
{
    int class_id{-1};
    float confidence;
    cv::Rect2f bbox;
    std::string class_name{};

    // MOT specific
    int frame{-1};
    int id{-1};
    cv::Point3f position{0.0f, 0.0f, 0.0f};

    // Reid specific
    std::vector<float> features{};

    cv::Scalar getColor() const
    {
        srand(class_id);
        return cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    }

    // For MOT file I/O
    friend std::istream &operator>>(std::istream &is, Detection &detection)
    {
        std::string field;

        std::getline(is, field, ',');
        detection.frame = std::stoi(field);

        std::getline(is, field, ',');
        detection.id = std::stoi(field);

        std::getline(is, field, ',');
        detection.bbox.x = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.y = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.width = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.height = std::stof(field);

        std::getline(is, field, ',');
        detection.confidence = std::stof(field);

        std::getline(is, field, ',');
        detection.position.x = std::stof(field);

        std::getline(is, field, ',');
        detection.position.y = std::stof(field);

        std::getline(is, field);
        detection.position.z = std::stof(field);

        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const Detection &detection)
    {
        os << detection.frame << "," << detection.id << ","
           << detection.bbox.x << "," << detection.bbox.y << ","
           << detection.bbox.width << "," << detection.bbox.height << ","
           << detection.confidence << ","
           << detection.position.x << "," << detection.position.y << "," << detection.position.z;
        return os;
    }
};