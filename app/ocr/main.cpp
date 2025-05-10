#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <types/detection.hpp>
#include <types/frame.hpp>
#include <models/ocr/ppocr.hpp>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description options("Program options");
    options.add_options()("help,h", "Show help message");
    options.add_options()("input,i", po::value<std::string>()->required(), "Input image");
    options.add_options()("config,c", po::value<std::string>(), "Path to model config.json");
    options.add_options()("display,d", po::bool_switch(), "Display image with results");
    options.add_options()("output,o", po::value<std::string>(), "Output JSON file for results");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);

    if (vm.count("help"))
    {
        std::cout << options << "\n";
        return 1;
    }

    po::notify(vm);

    // Input
    std::string imagePath = vm["input"].as<std::string>();
    Frame frame(cv::imread(imagePath, cv::IMREAD_COLOR));
    if (frame.empty())
    {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return 1;
    }

    // Config
    std::string configPath = vm["config"].as<std::string>();
    auto config = ocr::PPOCRConfig();
    config.loadFromJson(nlohmann::json::parse(std::ifstream(configPath)));

    // Process image
    ocr::PPOCRV3Recognizer recognizer(config);
    auto text = recognizer.process(frame.image);

    std::cout << text << std::endl;

    return 0;
}