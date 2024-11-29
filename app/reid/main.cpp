#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <utils/detection_utils.hpp>
#include <utils/vector_utils.hpp>

#include "reid.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[])
{

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")("input,i", po::value<std::string>()->required(), "Input image")("config,c", po::value<std::string>(), "Path to model config");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    po::notify(vm);

    // Input
    std::string imagePath = vm["input"].as<std::string>();
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return 1;
    }

    // Config
    ReIdConfig config;
    if (vm.count("config"))
    {
        std::string configPath = vm["config"].as<std::string>();
        config = ReIdConfig::load(configPath);
    }

    // Load model
    ReId reid(config);
    std::vector<float> featureVector;
    reid.process(image, featureVector);

    std::cout << "SIZE : " << featureVector.size() << std::endl;
    std::cout << "NORM : " << vector_ops::dot(featureVector, featureVector) << std::endl;

    return 0;
}