#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <engine/processor.hpp>
#include <models/detection/detector.hpp>
#include <utils/detection_utils.hpp>
#include <utils/vector_utils.hpp>
#include <opencv2/opencv.hpp>

class Segmentator : public trt::SIMOProcessor<std::vector<Detection>>
{
public:
    Segmentator(const DetectorConfig &t_config)
        : trt::SIMOProcessor<std::vector<Detection>>(t_config.engine), config(t_config) {};
    virtual ~Segmentator() = default;
    const DetectorConfig &getConfig() const { return config; };
    const std::string getClassName(int class_id) const
    {
        return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
    };

private:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    std::vector<Detection> postprocess(const trt::MultiOutput &features) override;

protected:
    const DetectorConfig config;
};

bool Segmentator::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
{
    std::cout << "Original frame sum: " << cv::sum(srcImg) << std::endl;

    // The model expects RGB input
    cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);
    std::cout << "After BGR2RGB sum: " << cv::sum(dstImg) << std::endl;

    // Resize the model to the expected size and pad with background
    dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);
    std::cout << "After letterbox sum: " << cv::sum(dstImg) << std::endl;

    // Convert to Float32
    dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
    std::cout << "After normalization sum: " << cv::sum(dstImg) << std::endl;

    return !dstImg.empty();
}

std::vector<Detection> Segmentator::postprocess(const trt::MultiOutput &features)
{
    std::cout << "# Outputs: " << features.size() << std::endl;
    std::cout << "Output 1: " << features[0].size() << " signature: " << vector_ops::sum(features[0]) << std::endl;
    std::cout << "Output 2: " << features[1].size() << " signature: " << vector_ops::sum(features[1]) << std::endl;

    std::vector<Detection> outputs{};
    return outputs;
}

int main()
{

    std::unique_ptr<Segmentator> model = nullptr;
    std::ifstream file("data/config.json");
    auto data = nlohmann::json::parse(file);

    auto config = DetectorConfig();
    config.loadFromJson(data["detector"]);
    std::vector<cv::Mat> batch;

    cv::Mat frame1 = cv::imread("data/zidane1.jpg", cv::IMREAD_COLOR);
    cv::Mat frame2 = cv::imread("data/zidane2.jpg", cv::IMREAD_COLOR);

    batch.push_back(frame1);
    batch.push_back(frame2);

    model = std::make_unique<Segmentator>(config);
    model->process(batch);
}