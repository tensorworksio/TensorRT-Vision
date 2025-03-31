#include <string>
#include <signal.h>
#include <atomic>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include <types/frame.hpp>
#include <tracking/factory.hpp>
#include <models/reid/reid.hpp>
#include <models/detection/factory.hpp>
#include <models/segmentation/factory.hpp>

namespace po = boost::program_options;

std::atomic<bool> running{true};

void signalHandler([[maybe_unused]] int signum)
{
    running = false;
}

int main(int argc, char *argv[])
{
    po::options_description options("Program options");
    options.add_options()("help,h", "Show help message");
    options.add_options()("input,i", po::value<std::string>()->required(), "Input video file or camera index (0,1,...)");
    options.add_options()("config,c", po::value<std::string>(), "Path to model config.json");
    options.add_options()("reid", po::bool_switch(), "Activate ReId");
    options.add_options()("output,o", po::value<std::string>(), "Output video file");
    options.add_options()("display,d", po::bool_switch(), "Display video frames");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);

    if (vm.count("help"))
    {
        std::cout << options << "\n";
        return 1;
    }

    po::notify(vm);

    // Input
    std::string inputPath = vm["input"].as<std::string>();
    cv::VideoCapture cap;
    if (inputPath.size() == 1 && std::isdigit(inputPath[0]))
    {
        cap.open(std::stoi(inputPath));
    }
    else
    {
        cap.open(inputPath);
    }
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video source " << inputPath << std::endl;
        return 1;
    }

    // Load config
    std::string configPath = vm["config"].as<std::string>();
    std::ifstream file(configPath);
    auto config = nlohmann::json::parse(file);
    bool reid = vm["reid"].as<bool>() && config.contains("reid");
    bool segment = config.contains("segmenter");

    // Load tracker & model
    auto tracker = TrackerFactory::create(configPath);

    std::unique_ptr<reid::ReId> reidModel = nullptr;
    if (reid)
    {
        auto reidConfig = reid::ReIdConfig::load(configPath, "reid");
        reidModel = std::make_unique<reid::ReId>(reidConfig);
    }

    std::unique_ptr<trt::DetectionProcessor> detector = nullptr;
    if (segment)
    {
        detector = seg::SegmenterFactory::create(configPath);
    }
    else
    {
        detector = det::DetectorFactory::create(configPath);
    }

    // Output
    cv::VideoWriter writer;
    if (vm.count("output"))
    {
        std::string outputPath = vm["output"].as<std::string>();
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                           cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(outputPath, fourcc, fps, frameSize);
        if (!writer.isOpened())
        {
            std::cerr << "Error: Could not create output video " << outputPath << std::endl;
            return 1;
        }
    }

    // Display
    bool display = vm["display"].as<bool>() || !vm.count("output");
    if (display)
    {
        cv::namedWindow("Multi Object Tracking", cv::WINDOW_AUTOSIZE);
    }

    Frame frame;
    signal(SIGINT, signalHandler);

    while (running)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Detect objects
        auto detections = detector->process(frame.image);

        // Extract features for each detection
        if (reidModel)
        {
            for (auto &det : detections)
            {
                cv::Mat roi = frame.image(det.bbox);
                det.features = reidModel->process(roi);
            }
        }

        // Update tracker
        tracker->update(detections);

        // Visualize results
        cv::Mat output = frame.draw(detections, true, true);

        if (display)
            cv::imshow("Multi Object Tracking", output);

        if (writer.isOpened())
            writer.write(output);

        if (cv::waitKey(1) == 27)
            running = false;
    }

    // Cleanup
    if (cap.isOpened())
        cap.release();

    if (writer.isOpened())
        writer.release();

    if (display)
        cv::destroyAllWindows();

    return 0;
}