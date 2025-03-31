#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <utils/geometry_utils.hpp>
#include <models/reid/reid.hpp>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description options("Program options");
    options.add_options()("help,h", "Show help message");
    options.add_options()("query,q", po::value<std::string>()->required(), "Query image to compare");
    options.add_options()("key,k", po::value<std::string>()->required(), "Key image to compare against");
    options.add_options()("config,c", po::value<std::string>(), "Path to model config.json");
    options.add_options()("output,o", po::value<std::string>(), "Output file");
    options.add_options()("display,d", po::bool_switch(), "Display images");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);

    if (vm.count("help"))
    {
        std::cout << options << "\n";
        return 1;
    }

    po::notify(vm);

    // Input query
    std::string queryPath = vm["query"].as<std::string>();
    cv::Mat queryImage = cv::imread(queryPath, cv::IMREAD_COLOR);
    if (queryImage.empty())
    {
        std::cerr << "Error: Could not load query image " << queryPath << std::endl;
        return 1;
    }

    // Input key
    std::string keyPath = vm["key"].as<std::string>();
    cv::Mat keyImage = cv::imread(keyPath, cv::IMREAD_COLOR);
    if (keyImage.empty())
    {
        std::cerr << "Error: Could not load key image " << keyPath << std::endl;
        return 1;
    }

    // Config
    reid::ReIdConfig config;
    std::string configPath = vm["config"].as<std::string>();
    config = reid::ReIdConfig::load(configPath);

    // Process images
    reid::ReId reid(config);
    auto featureVector1 = reid.process(queryImage);
    auto featureVector2 = reid.process(keyImage);

    // Output
    float similarity = cosineSimilarity(featureVector1, featureVector2);
    bool match = similarity > reid.getConfig().confidenceThreshold;
    nlohmann::json output = {
        {"status", "success"},
        {"data", {
                     {"match", match},
                     {"similarity", similarity},
                 }}};

    // Output results
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