#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <types/detection.hpp>
#include <models/classification/classifier.hpp>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description options("Program options");
    options.add_options()("help,h", "Show help message");
    options.add_options()("input,i", po::value<std::string>()->required(), "Input image");
    options.add_options()("config,c", po::value<std::string>(), "Path to model config.json");
    options.add_options()("display,d", po::bool_switch(), "Display image with results");
    options.add_options()("output,o", po::value<std::string>(), "Output text file for results");

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
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return 1;
    }

    // Config
    std::string configPath = vm["config"].as<std::string>();
    auto config = cls::ClassifierConfig::load(configPath);

    // Process image
    cls::Classifier classifier(config);
    Detection det = classifier.process(image);

    // Output
    nlohmann::json output = {
        {"status", "success"},
        {"data", {{"class_id", det.class_id}, {"class_name", det.class_name}, {"confidence", det.confidence}}}};

    if (vm.count("output"))
    {
        std::string outputPath = vm["output"].as<std::string>();
        std::ofstream outFile(outputPath);
        if (outFile.is_open())
        {
            outFile << output.dump() << std::endl;
            outFile.close();
        }
        else
        {
            nlohmann::json error = {
                {"status", "error"},
                {"message", "Could not create output file"}};
            std::cerr << error.dump() << std::endl;
            return 1;
        }
    }
    else
    {
        // Write to stdout if no output file specified
        std::cout << output.dump() << std::endl;
    }

    // Display image if requested
    if (vm["display"].as<bool>())
    {
        cv::putText(image,
                    det.class_name + " (" + std::to_string(det.confidence) + ")",
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