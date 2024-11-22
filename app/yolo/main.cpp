#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

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

    return 0;
}