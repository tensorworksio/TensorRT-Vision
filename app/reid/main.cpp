#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <utils/detection_utils.hpp>
#include "reid.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")("query,q", po::value<std::string>()->required(), "Query image to compare")("key,k", po::value<std::string>()->required(), "Key image to compare against")("config,c", po::value<std::string>(), "Path to model config")("output,o", po::value<std::string>(), "Output file (optional)")("display,d", po::bool_switch(), "Display images side by side");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    po::notify(vm);

    // Load query image
    std::string queryPath = vm["query"].as<std::string>();
    cv::Mat queryImage = cv::imread(queryPath, cv::IMREAD_COLOR);
    if (queryImage.empty())
    {
        std::cerr << "Error: Could not load query image " << queryPath << std::endl;
        return 1;
    }

    // Load key image
    std::string keyPath = vm["key"].as<std::string>();
    cv::Mat keyImage = cv::imread(keyPath, cv::IMREAD_COLOR);
    if (keyImage.empty())
    {
        std::cerr << "Error: Could not load key image " << keyPath << std::endl;
        return 1;
    }

    // Config
    ReIdConfig config;
    if (vm.count("config"))
    {
        std::string configPath = vm["config"].as<std::string>();
        config = ReIdConfig::load(configPath);
    }

    // Load model and process images
    ReId reid(config);
    auto featureVector1 = reid.process(queryImage);
    auto featureVector2 = reid.process(keyImage);

    // Compute cosine similarity
    double similarity = cosineSimilarity(featureVector1, featureVector2);

    // Output results
    if (vm.count("output"))
    {
        std::string outputPath = vm["output"].as<std::string>();
        std::ofstream outFile(outputPath);
        if (outFile.is_open())
        {
            outFile << "Cosine Similarity: " << similarity << std::endl;
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
        std::cout << "Cosine Similarity: " << similarity << std::endl;
    }

    // Display images if requested
    if (vm["display"].as<bool>())
    {
        int maxHeight = std::max(queryImage.rows, keyImage.rows);
        int totalWidth = queryImage.cols + keyImage.cols;

        cv::Mat canvas(maxHeight, totalWidth, CV_8UC3);
        cv::Mat leftROI(canvas, cv::Rect(0, 0, queryImage.cols, queryImage.rows));
        cv::Mat rightROI(canvas, cv::Rect(queryImage.cols, 0, keyImage.cols, keyImage.rows));

        queryImage.copyTo(leftROI);
        keyImage.copyTo(rightROI);

        cv::putText(canvas, "Query", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(canvas, "Key", cv::Point(queryImage.cols + 10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::namedWindow("ReID Comparison", cv::WINDOW_AUTOSIZE);
        cv::imshow("ReID Comparison", canvas);
        cv::waitKey(0);
    }

    return 0;
}