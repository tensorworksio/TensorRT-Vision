#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <models/segmentation/yolo.hpp>

int main()
{

    std::ifstream file("data/config.json");
    auto data = nlohmann::json::parse(file);
    seg::YoloConfig config;
    config.loadFromJson(data["detector"]);

    seg::Yolo model = seg::Yolo(config);
    cv::Mat frame = cv::imread("data/zidane2.jpg", cv::IMREAD_COLOR);

    std::vector<Detection> detections;
    detections = model.process(frame);

    if (detections.empty())
    {
        std::cout << "No detections found." << std::endl;
    }
    else
    {
        std::cout << "Found " << detections.size() << " detections:" << std::endl;
        for (const auto &det : detections)
        {
            std::cout << "Class: " << det.class_name
                      << ", Confidence: " << det.confidence
                      << ", Box: [" << det.bbox.x << ", " << det.bbox.y
                      << ", " << det.bbox.width << ", " << det.bbox.height << "]"
                      << std::endl;
        }
    }

    for (const auto &detection : detections)
    {
        if (detection.mask.empty())
            continue;

        // Draw mask
        cv::Mat fullMask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        detection.mask.copyTo(fullMask(detection.bbox));

        cv::Mat overlay = frame.clone();
        overlay.setTo(detection.getClassColor(), fullMask);

        double alpha = 0.5;
        cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame);

        // Draw bounding box
        cv::rectangle(frame, detection.bbox, detection.getClassColor(), 2);

        // Draw label
        std::string label = detection.class_name + " " + std::to_string(detection.confidence).substr(0, 4);
        cv::putText(frame, label, cv::Point(detection.bbox.x, detection.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, detection.getClassColor(), 2);
    }
    cv::imshow("Segmentation", frame);
    cv::waitKey(0);
}