#include <string>
#include <signal.h>
#include <atomic>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>
#include <models/reid/reid.hpp>
#include <models/detection/factory.hpp>

#include <tracking/factory.hpp>

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

    // Load tracker & detector
    std::string configPath = vm["config"].as<std::string>();
    auto tracker = TrackerFactory::create(configPath);
    auto detector = DetectorFactory::create(configPath);
    std::unique_ptr<ReId> reidModel = nullptr;

    // Load reid
    std::ifstream file(configPath);
    auto config = nlohmann::json::parse(file);
    bool reid = vm["reid"].as<bool>() && config.contains("reid");

    if (reid)
    {
        auto reidConfig = ReIdConfig::load(configPath, "reid");
        reidModel = std::make_unique<ReId>(reidConfig);
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

    cv::Mat frame;
    signal(SIGINT, signalHandler);

    while (running)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Detect objects
        auto detections = detector->process(frame);

        // Extract features for each detection
        if (reidModel)
        {
            for (auto &det : detections)
            {
                cv::Mat roi = frame(det.bbox);
                det.features = reidModel->process(roi);
            }
        }

        // Update tracker
        tracker->update(detections);

        // Visualize results
        for (const auto &det : detections)
        {
            cv::rectangle(frame, det.bbox, det.getTrackColor(), 2);
            std::string label = det.class_name + " ID:" + std::to_string(det.id);
            cv::putText(frame, label,
                        cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, det.getTrackColor(), 2);
        }

        if (display)
        {
            cv::imshow("Multi Object Tracking", frame);
        }

        if (writer.isOpened())
        {
            writer.write(frame);
        }

        if (cv::waitKey(1) == 27)
        {
            running = false;
        }
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