#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <types/detection.hpp>
#include "classifier.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")("input,i", po::value<std::string>()->required(), "Input image")("config,c", po::value<std::string>(), "Path to model config")("display,d", po::bool_switch(), "Display image with results")("output,o", po::value<std::string>(), "Output text file for results");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    po::notify(vm);

    bool display = vm["display"].as<bool>();

    // Input
    std::string imagePath = vm["input"].as<std::string>();
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return 1;
    }

    // Config
    ClassifierConfig config;
    if (vm.count("config"))
    {
        std::string configPath = vm["config"].as<std::string>();
        config = ClassifierConfig::load(configPath);
    }

    // Process image
    Detection det;
    Classifier classifier(config);
    det = classifier.process(image);

    // Write to output file if specified
    if (vm.count("output"))
    {
        std::string outputPath = vm["output"].as<std::string>();
        std::ofstream outFile(outputPath);
        if (outFile.is_open())
        {
            outFile << "Category: " << det.class_name << std::endl;
            outFile << "Confidence: " << det.probability << std::endl;
            outFile.close();
        }
        else
        {
            std::cerr << "Error: Could not create output file " << outputPath << std::endl;
            return 1;
        }
    }
    else
    {
        // Write to stdout if no output file specified
        std::cout << "Category: " << det.class_name << std::endl;
        std::cout << "Confidence: " << det.probability << std::endl;
    }

    // Display image if requested
    if (display)
    {
        cv::putText(image,
                    det.class_name + " (" + std::to_string(det.probability) + ")",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2);

        cv::namedWindow("Classification Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Classification Result", image);
        cv::waitKey(0);
    }

    return 0;
}