#include <fstream>
#include <boost/filesystem.hpp>
#include "engine/engine.hpp"
#include "engine/utils.hpp"

namespace fs = boost::filesystem;

namespace trt
{

    const std::string toString(Precision precision)
    {
        switch (precision)
        {
        case Precision::INT8:
            return "INT8";
        case Precision::FP16:
            return "FP16";
        case Precision::FP32:
        default:
            return "FP32";
        }
    }

    Engine::Engine(const EngineOptions &options, NvLogger logger) : m_options(options), m_logger(logger) {}

    Engine::~Engine()
    {
        clearBuffers();
        m_context.reset();
        m_engine.reset();
        m_runtime.reset();
    }

    void Engine::clearBuffers()
    {
        for (auto &buffer : m_buffers)
        {
            cuda::checkCudaErrorCode(cudaFree(buffer));
        }

        m_buffers.clear();
        m_outputLengths.clear();
        m_inputDims.clear();
        m_outputDims.clear();
        m_IOTensorNames.clear();
    }

    bool Engine::loadNetwork(const std::string &engineModelPath)
    {
        // Read serialized model from disk
        if (!fs::exists(engineModelPath))
        {
            m_logger.log(NvLogger::Severity::kERROR, "{} does not exist", engineModelPath);
            return false;
        }

        std::ifstream file(engineModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
        {
            m_logger.log(NvLogger::Severity::kERROR, "Failed to read engine model from disk");
            return false;
        }

        // Create a runtime
        m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
        if (m_runtime == nullptr)
        {
            m_logger.log(NvLogger::Severity::kERROR, "Failed to create InferRuntime");
            return false;
        }

        // Set device
        cuda::checkCudaErrorCode(cudaSetDevice(m_options.deviceIndex));

        // Create engine
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (m_engine == nullptr)
        {
            m_logger.log(NvLogger::Severity::kERROR, "Failed to deserialize engine");
            return false;
        }

        // Create execution context
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (m_context == nullptr)
        {
            m_logger.log(NvLogger::Severity::kERROR, "Failed to create execution context");
            return false;
        }

        // Create CUDA stream
        cudaStream_t stream;
        cuda::checkCudaErrorCode(cudaStreamCreate(&stream));

        // Allocate GPU memory for input and output buffers
        clearBuffers();
        m_buffers.resize(m_engine->getNbIOTensors());

        for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
        {
            const auto tensorName = m_engine->getIOTensorName(i);
            const auto tensorType = m_engine->getTensorIOMode(tensorName);
            const auto tensorShape = m_engine->getTensorShape(tensorName);
            const auto tensorDataType = m_engine->getTensorDataType(tensorName);
            m_IOTensorNames.emplace_back(tensorName);

            if (tensorDataType != nvinfer1::DataType::kFLOAT)
            {
                m_logger.log(NvLogger::Severity::kERROR, "Only FLOAT32 is supported for inputs/outputs");
                return false;
            }

            if (tensorType == nvinfer1::TensorIOMode::kINPUT)
            {
                uint32_t inputMemSize = m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float);
                cuda::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], inputMemSize, stream));
                // TODO: deal with input of any dim
                m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            }
            else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT)
            {
                uint32_t outputLength = 1;
                m_outputDims.push_back(tensorShape);
                for (int j = 1; j < tensorShape.nbDims; ++j)
                {
                    // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                    outputLength *= tensorShape.d[j];
                }
                m_outputLengths.push_back(outputLength);
                uint32_t outputMemSize = m_options.maxBatchSize * outputLength * sizeof(float);
                cuda::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputMemSize, stream));
            }
            else
            {
                m_logger.log(NvLogger::Severity::kERROR, "IO Tensor {} is neither kINPUT nor kOUTPUT", tensorName);
                return false;
            }
        }

        // Synchronize and destroy the CUDA stream
        cuda::checkCudaErrorCode(cudaStreamSynchronize(stream));
        cuda::checkCudaErrorCode(cudaStreamDestroy(stream));

        return true;
    }

    bool Engine::prepareInputs(const std::vector<std::vector<cv::Mat>> &inputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize)
    {
        const auto numInputs = m_inputDims.size();

        for (size_t i = 0; i < numInputs; ++i)
        {
            const auto &dims = m_inputDims[i];
            const auto &inputBatch = inputs[i];

            auto &input = inputBatch[0];
            if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2])
            {
                m_logger.log(NvLogger::Severity::kERROR, "Input does not have correct size!");
                m_logger.log(NvLogger::Severity::kERROR, "Expected: ({}, {}, {})", dims.d[0], dims.d[1], dims.d[2]);
                m_logger.log(NvLogger::Severity::kERROR, "Got: ({}, {}, {})", input.channels(), input.rows, input.cols);
                m_logger.log(NvLogger::Severity::kERROR, "Ensure you resize your input image to the correct size.");
                return false;
            }

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            // TODO: Separate m_InputTensor and m_OutputTensors
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);
            // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format
            auto mfloat = blobFromMats(inputBatch);
            auto *dataPointer = mfloat.ptr<void>();

            cuda::checkCudaErrorCode(cudaMemcpyAsync(
                m_buffers[i], dataPointer, mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float), cudaMemcpyHostToDevice, inferenceCudaStream));
        }
        return true;
    }

    bool Engine::runInference(const cv::Mat image, std::vector<float> &featureVector)
    {
        // Single batch SISO inference (SBSISO)

        // Convert the input image into a batch of size 1
        std::vector<cv::Mat> input_batch(1, image);

        // Call Multi batch SISO (MBSISO)
        std::vector<std::vector<float>> output_batch;
        bool success = runInference(input_batch, output_batch);

        // Extract the feature vector from the MBSISO output
        if (success)
            featureVector = output_batch[0];
        return success;
    }

    bool Engine::runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<float>> &outputBatch)
    {
        // Multi batch SISO inference (MBSISO)

        // Convert the input batch into a batch vector of size 1
        std::vector<std::vector<cv::Mat>> inputs(1, inputBatch);

        // Call MIMO function
        std::vector<std::vector<std::vector<float>>> outputs;
        bool success = runInference(inputs, outputs);

        // Extract the first output batch from the MIMO result
        if (success)
            std::transform(
                outputs.begin(), outputs.end(), std::back_inserter(outputBatch), [](const std::vector<std::vector<float>> &output)
                { return output.front(); });
        return success;
    }

    bool Engine::runInference(const std::vector<std::vector<cv::Mat>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs)
    {
        // MIMO inference
        if (inputs.empty() || inputs[0].empty())
        {
            m_logger.log(NvLogger::Severity::kERROR, "Provided input vector is empty!");
            return false;
        }

        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs)
        {
            m_logger.log(NvLogger::Severity::kERROR, "Incorrect number of inputs provided!");
            m_logger.log(NvLogger::Severity::kERROR, "Expected {} inputs, got {}", numInputs, inputs.size());
            return false;
        }

        // Ensure the batch size does not exceed the max
        if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize))
        {
            m_logger.log(NvLogger::Severity::kERROR, "The batch size is larger than the model expects!");
            m_logger.log(NvLogger::Severity::kERROR, "Expected batch of size {}, got {}", m_options.maxBatchSize, inputs[0].size());
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        // Make sure the same batch size was provided for all inputs
        for (size_t i = 1; i < inputs.size(); ++i)
        {
            if (inputs[i].size() != static_cast<size_t>(batchSize))
            {
                m_logger.log(NvLogger::Severity::kERROR, "The batch size needs to be constant for all inputs!");
                m_logger.log(NvLogger::Severity::kERROR, "Expected batch of size {}, got {}", m_options.maxBatchSize, inputs[i].size());
                return false;
            }
        }

        // Create the cuda stream that will be used for inference
        cudaStream_t inferenceCudaStream;
        cuda::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        bool status; // Used for error checking

        // Load inputs to CUDA memory
        status = prepareInputs(inputs, inferenceCudaStream, batchSize);
        if (!status)
        {
            return false;
        }

        // Ensure all dynamic bindings have been defined
        if (!m_context->allInputDimensionsSpecified())
        {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }

        // Set the address of the input and output buffers
        for (size_t i = 0; i < m_buffers.size(); ++i)
        {
            status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
            if (!status)
            {
                return false;
            }
        }

        // Run inference
        status = m_context->enqueueV3(inferenceCudaStream);
        if (!status)
        {
            return false;
        }

        // Copy the outputs back to CPU
        status = prepareOutputs(outputs, inferenceCudaStream, batchSize);
        if (!status)
        {
            return false;
        }
        // Synchronize the cuda stream
        cuda::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        cuda::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
        return true;
    }

    bool Engine::prepareOutputs(std::vector<std::vector<std::vector<float>>> &outputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize)
    {
        outputs.clear();
        const auto numInputs = m_inputDims.size();
        for (int batch = 0; batch < batchSize; ++batch)
        {
            // Batch
            std::vector<std::vector<float>> batchOutputs{};
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding)
            {
                // TODO: just separate inputs/outputs in different buffers
                // We start at index m_inputDims.size() to account for the inputs in our m_buffers
                std::vector<float> output;
                auto outputLength = m_outputLengths[outputBinding - numInputs];
                output.resize(outputLength);
                // Copy the output
                cuda::checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                                         static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLength),
                                                         outputLength * sizeof(float),
                                                         cudaMemcpyDeviceToHost,
                                                         inferenceCudaStream));
                batchOutputs.emplace_back(std::move(output));
            }
            outputs.emplace_back(std::move(batchOutputs));
        }
        return true;
    }

    void setEngineOptions(EngineOptions &options, int batchSize, Precision precision)
    {
        // Specify what precision to use for inference. FP16 is approximately twice as fast as FP32.
        options.precision = precision;
        // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
        // Specify the batch size to optimize for.
        options.optBatchSize = batchSize;
        // Specify the maximum batch size we plan on running.
        options.maxBatchSize = batchSize;
    }

} // namespace trt