#include <fstream>
#include <filesystem>
#include "engine/engine.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/tensorrt_utils.hpp"

namespace fs = std::filesystem;

namespace trt
{

    Engine::Engine(const EngineOptions &options) : m_options(options) {}

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

    void Engine::loadNetwork(const std::string &engineModelPath)
    {
        // Read serialized model from disk
        if (!fs::exists(engineModelPath))
            throw std::runtime_error("Engine model not found: " + engineModelPath);

        std::ifstream file(engineModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
            throw std::runtime_error("Failed to read engine model from disk: " + engineModelPath);

        // Create a runtime
        m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
        if (!m_runtime)
            throw std::runtime_error("Failed to create InferRuntime");

        // Set device
        cuda::checkCudaErrorCode(cudaSetDevice(m_options.deviceIndex));

        // Create engine
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!m_engine)
            throw std::runtime_error("Failed to deserialize engine: " + engineModelPath);

        // Create execution context
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context)
            throw std::runtime_error("Failed to create execution context");

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
                throw std::runtime_error("Only FLOAT32 is supported for inputs/outputs");

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
                throw std::runtime_error(std::string("IO tensor is neither kINPUT nor kOUTPUT: ") + tensorName);
            }
        }

        // Synchronize and destroy the CUDA stream
        cuda::checkCudaErrorCode(cudaStreamSynchronize(stream));
        cuda::checkCudaErrorCode(cudaStreamDestroy(stream));
    }

    void Engine::prepareInputs(const std::vector<std::vector<cv::Mat>> &inputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize)
    {
        const auto numInputs = m_inputDims.size();

        for (size_t i = 0; i < numInputs; ++i)
        {
            const auto &dims = m_inputDims[i];
            const auto &inputBatch = inputs[i];

            auto &input = inputBatch[0];
            if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2])
                throw std::runtime_error(
                    "Input size mismatch: expected (" +
                    std::to_string(dims.d[0]) + ", " + std::to_string(dims.d[1]) + ", " + std::to_string(dims.d[2]) +
                    "), got (" +
                    std::to_string(input.channels()) + ", " + std::to_string(input.rows) + ", " + std::to_string(input.cols) + ")");

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            // TODO: Separate m_InputTensor and m_OutputTensors
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);
            // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format
            auto mfloat = blobFromMats(inputBatch);
            auto *dataPointer = mfloat.ptr<void>();

            cuda::checkCudaErrorCode(cudaMemcpyAsync(
                m_buffers[i], dataPointer, mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float), cudaMemcpyHostToDevice, inferenceCudaStream));
        }
    }

    void Engine::runInference(const cv::Mat &image, std::vector<float> &featureVector)
    {
        // Single batch SISO inference (SBSISO)
        std::vector<cv::Mat> input_batch(1, image);
        std::vector<std::vector<float>> output_batch;
        runInference(input_batch, output_batch);
        featureVector = output_batch[0];
    }

    void Engine::runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<float>> &outputBatch)
    {
        // Multi batch SISO inference (MBSISO)
        std::vector<std::vector<cv::Mat>> inputs(1, inputBatch);
        std::vector<std::vector<std::vector<float>>> outputs;
        runInference(inputs, outputs);
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputBatch),
                       [](const std::vector<std::vector<float>> &output) { return output.front(); });
    }

    void Engine::runInference(const cv::Mat &image, std::vector<std::vector<float>> &outputs)
    {
        // Single batch SIMO inference (SBSIMO)
        std::vector<cv::Mat> input_batch(1, image);
        std::vector<std::vector<std::vector<float>>> output_batch;
        runInference(input_batch, output_batch);
        outputs = output_batch[0];
    }

    void Engine::runInference(const std::vector<cv::Mat> &inputBatch, std::vector<std::vector<std::vector<float>>> &outputBatch)
    {
        // Multi batch SIMO inference (MBSIMO)
        std::vector<std::vector<cv::Mat>> inputs(1, inputBatch);
        runInference(inputs, outputBatch);
    }

    void Engine::runInference(const std::vector<std::vector<cv::Mat>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs)
    {
        // Multi batch MIMO inference (MBMIMO)
        if (inputs.empty() || inputs[0].empty())
            throw std::runtime_error("Provided input vector is empty");

        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs)
            throw std::runtime_error("Incorrect number of inputs: expected " +
                                     std::to_string(numInputs) + ", got " + std::to_string(inputs.size()));

        if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize))
            throw std::runtime_error("Batch size " + std::to_string(inputs[0].size()) +
                                     " exceeds max batch size " + std::to_string(m_options.maxBatchSize));

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        for (size_t i = 1; i < inputs.size(); ++i)
        {
            if (inputs[i].size() != static_cast<size_t>(batchSize))
                throw std::runtime_error("Inconsistent batch sizes across inputs");
        }

        cudaStream_t inferenceCudaStream;
        cuda::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        prepareInputs(inputs, inferenceCudaStream, batchSize);

        if (!m_context->allInputDimensionsSpecified())
            throw std::runtime_error("Not all required input dimensions specified");

        for (size_t i = 0; i < m_buffers.size(); ++i)
        {
            if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]))
                throw std::runtime_error("Failed to set tensor address for: " + m_IOTensorNames[i]);
        }

        if (!m_context->enqueueV3(inferenceCudaStream))
            throw std::runtime_error("Failed to run inference");

        prepareOutputs(outputs, inferenceCudaStream, batchSize);

        cuda::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        cuda::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    }

    void Engine::prepareOutputs(std::vector<std::vector<std::vector<float>>> &outputs, cudaStream_t &inferenceCudaStream, const int32_t batchSize)
    {
        outputs.clear();
        const auto numInputs = m_inputDims.size();
        for (int batch = 0; batch < batchSize; ++batch)
        {
            std::vector<std::vector<float>> batchOutputs{};
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding)
            {
                // TODO: just separate inputs/outputs in different buffers
                // We start at index m_inputDims.size() to account for the inputs in our m_buffers
                std::vector<float> output;
                auto outputLength = m_outputLengths[outputBinding - numInputs];
                output.resize(outputLength);
                cuda::checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                                         static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLength),
                                                         outputLength * sizeof(float),
                                                         cudaMemcpyDeviceToHost,
                                                         inferenceCudaStream));
                batchOutputs.emplace_back(std::move(output));
            }
            outputs.emplace_back(std::move(batchOutputs));
        }
    }

    void loadEngine(Engine &engine, const std::string &engineModelPath)
    {
        engine.loadNetwork(engineModelPath);
    }

    void setEngineOptions(EngineOptions &options, int batchSize, Precision precision)
    {
        // Specify what precision to use for inference. FP16 is approximately twice as fast as FP32
        options.precision = precision;
        // Specify the batch size to optimize for
        options.optBatchSize = batchSize;
        // Specify the maximum batch size we plan on running
        options.maxBatchSize = batchSize;
    }

} // namespace trt