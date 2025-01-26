#include <string>
#include <signal.h>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <models/detection/factory.hpp>

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

    // Load detector
    std::string configPath = vm["config"].as<std::string>();
    auto detector = DetectorFactory::create(configPath);

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
        cv::namedWindow("Detections", cv::WINDOW_AUTOSIZE);
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

        // Draw detections
        for (const auto &det : detections)
        {
            cv::rectangle(frame, det.bbox, det.getClassColor(), 2);
            cv::putText(frame, det.class_name,
                        cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, det.getClassColor(), 2);
        }

        if (display)
        {
            cv::imshow("Detections", frame);
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

    if (cap.isOpened())
        cap.release();

    if (writer.isOpened())
        writer.release();

    if (display)
        cv::destroyAllWindows();

    return 0;
}