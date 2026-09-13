#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "FixedAddressArena.h"
#include "glow/Base/InputZeroDither.h"
#include "glow/Support/WritebackPatchSeedRuntime.h"

#if defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_MNIST)
#include "lenet_mnist_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY lenet_mnist_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_NAME "relu_off_maxpool_off"
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_MNIST)
#include "lenet_mnist_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY lenet_mnist_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_NAME "relu_on_maxpool_off"
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_MNIST)
#include "lenet_mnist_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY lenet_mnist_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_NAME "relu_on_maxpool_on"
#elif defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_MNIST64)
#include "lenet_mnist64_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY lenet_mnist64_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST64_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_NAME "relu_off_maxpool_off"
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_MNIST64)
#include "lenet_mnist64_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY lenet_mnist64_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST64_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST64_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST64_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_NAME "relu_on_maxpool_off"
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_MNIST64)
#include "lenet_mnist64_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY lenet_mnist64_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE LENET_MNIST64_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN LENET_MNIST64_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET LENET_MNIST64_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET LENET_MNIST64_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_NAME "relu_on_maxpool_on"
#else
#error "Define one bundle case macro and one bundle variant macro."
#endif

namespace {

constexpr size_t kInputN = 1;
constexpr size_t kInputC = 1;
#if defined(BUNDLE_VARIANT_MNIST)
constexpr size_t kInputH = 28;
constexpr size_t kInputW = 28;
#elif defined(BUNDLE_VARIANT_MNIST64)
constexpr size_t kInputH = 64;
constexpr size_t kInputW = 64;
#else
#error "Define one bundle variant macro."
#endif
constexpr size_t kInputBytes = kInputN * kInputC * kInputH * kInputW * sizeof(float);
constexpr size_t kOutputElements = 10;

void loadFileOrExit(const std::string &path, void *dest, size_t expectedSize) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::cerr << "failed to open: " << path << "\n";
    std::exit(1);
  }
  stream.read(reinterpret_cast<char *>(dest), expectedSize);
  if (static_cast<size_t>(stream.gcount()) != expectedSize) {
    std::cerr << "unexpected file size for " << path << ", expected "
              << expectedSize << " bytes\n";
    std::exit(1);
  }
}

void writeTextOutputOrExit(const std::string &path, const float *output,
                           size_t count) {
  std::ofstream stream(path);
  if (!stream) {
    std::cerr << "failed to write: " << path << "\n";
    std::exit(1);
  }
  stream << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (size_t index = 0; index < count; index++) {
    stream << output[index] << "\n";
  }
}

void writeBinaryOutputOrExit(const std::string &path, const float *output,
                             size_t count) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    std::cerr << "failed to write: " << path << "\n";
    std::exit(1);
  }
  stream.write(reinterpret_cast<const char *>(output), count * sizeof(float));
}

size_t argmax(const float *values, size_t count) {
  return static_cast<size_t>(
      std::distance(values, std::max_element(values, values + count)));
}

int runStreamMode(uint8_t *constantWeight, uint8_t *mutableWeight,
                  uint8_t *activations) {
  std::vector<char> inputBuffer(kInputBytes);
  const size_t outputBytes = kOutputElements * sizeof(float);

  while (true) {
    std::cin.read(inputBuffer.data(),
                  static_cast<std::streamsize>(inputBuffer.size()));
    const std::streamsize bytesRead = std::cin.gcount();
    if (bytesRead == 0) {
      return 0;
    }
    if (bytesRead != static_cast<std::streamsize>(inputBuffer.size())) {
      std::cerr << "unexpected stream payload size: got " << bytesRead
                << " bytes, expected " << inputBuffer.size() << "\n";
      return 2;
    }

    std::memcpy(mutableWeight + BUNDLE_DATA_OFFSET, inputBuffer.data(),
                inputBuffer.size());
    glow::applyInputZeroDitherRawOrExit(
        reinterpret_cast<float *>(mutableWeight + BUNDLE_DATA_OFFSET), kInputN,
        kInputC, kInputH, kInputW, glow::InputZeroDitherLayout::NCHW);

    glow::writebackpatchseed::resetInferencePatchSeed();
    const int rc = BUNDLE_ENTRY(constantWeight, mutableWeight, activations);
    if (rc != GLOW_SUCCESS) {
      std::cerr << "bundle returned error code: " << rc << "\n";
      return rc;
    }

    const auto *output =
        reinterpret_cast<const float *>(mutableWeight + BUNDLE_OUTPUT_OFFSET);
    std::cout.write(reinterpret_cast<const char *>(output),
                    static_cast<std::streamsize>(outputBytes));
    std::cout.flush();
    if (!std::cout) {
      std::cerr << "failed to write streamed output\n";
      return 1;
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  const bool streamMode = argc == 3 && std::string(argv[2]) == "--stream";
  if ((!streamMode && (argc < 3 || argc > 4)) || argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <weights.bin> <input_f32.bin> [output.txt]\n"
              << "   or: " << argv[0] << " <weights.bin> --stream\n";
    return 2;
  }

  const std::string weightsPath = argv[1];

  glow_vgg_onoff_mulsample::BundleMemoryArena arena(
      BUNDLE_MEM_ALIGN, BUNDLE_CONSTANT_MEM_SIZE, BUNDLE_MUTABLE_MEM_SIZE,
      BUNDLE_ACTIVATIONS_MEM_SIZE);
  auto *constantWeight = arena.constantWeight();
  auto *mutableWeight = arena.mutableWeight();
  auto *activations = arena.activations();

  loadFileOrExit(weightsPath, constantWeight, BUNDLE_CONSTANT_MEM_SIZE);
  if (streamMode) {
    return runStreamMode(constantWeight, mutableWeight, activations);
  }

  const std::string inputPath = argv[2];
  const std::string outputPath =
      argc >= 4 ? argv[3] : std::string("bundle_output_") + BUNDLE_NAME + ".txt";
  const std::string outputBinPath = outputPath + ".bin";

  loadFileOrExit(inputPath, mutableWeight + BUNDLE_DATA_OFFSET, kInputBytes);
  glow::applyInputZeroDitherRawOrExit(
      reinterpret_cast<float *>(mutableWeight + BUNDLE_DATA_OFFSET), kInputN,
      kInputC, kInputH, kInputW, glow::InputZeroDitherLayout::NCHW);

  glow::writebackpatchseed::resetInferencePatchSeed();
  const int rc = BUNDLE_ENTRY(constantWeight, mutableWeight, activations);
  if (rc != GLOW_SUCCESS) {
    std::cerr << "bundle returned error code: " << rc << "\n";
    return rc;
  }

  const auto *output =
      reinterpret_cast<const float *>(mutableWeight + BUNDLE_OUTPUT_OFFSET);
  writeTextOutputOrExit(outputPath, output, kOutputElements);
  writeBinaryOutputOrExit(outputBinPath, output, kOutputElements);

  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
  std::cout << "bundle=" << BUNDLE_NAME << "\n";
  std::cout << "output_path=" << outputPath << "\n";
  std::cout << "output_bin=" << outputBinPath << "\n";
  std::cout << "logits:";
  for (size_t index = 0; index < kOutputElements; index++) {
    std::cout << (index == 0 ? " " : ", ") << output[index];
  }
  std::cout << "\n";
  std::cout << "argmax=" << argmax(output, kOutputElements) << "\n";

  return 0;
}
