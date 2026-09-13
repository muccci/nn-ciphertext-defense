#include <cstdint>
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

#if defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_CIFAR)
#include "squeezenet1_0_cifar_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_CIFAR32)
#include "squeezenet1_0_cifar32_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar32_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR32_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_CIFAR)
#include "squeezenet1_0_cifar_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_CIFAR32)
#include "squeezenet1_0_cifar32_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar32_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_CIFAR)
#include "squeezenet1_0_cifar_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_CIFAR32)
#include "squeezenet1_0_cifar32_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY squeezenet1_0_cifar32_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_CIFAR32_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_OUTPUT_ELEMENTS 10
#elif defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_IMAGENET32)
#include "squeezenet1_0_imagenet32_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet32_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 1000
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_IMAGENET32)
#include "squeezenet1_0_imagenet32_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet32_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 1000
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_IMAGENET32)
#include "squeezenet1_0_imagenet32_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet32_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET32_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_OUTPUT_ELEMENTS 1000
#elif defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_IMAGENET100_224)
#include "squeezenet1_0_imagenet100_224_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet100_224_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET100_224_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 100
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_IMAGENET100_224)
#include "squeezenet1_0_imagenet100_224_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet100_224_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET100_224_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_OUTPUT_ELEMENTS 100
#elif defined(BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_IMAGENET50_96)
#include "squeezenet1_0_imagenet50_96_relu_off_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet50_96_relu_off_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_OFF_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 50
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_OFF) && defined(BUNDLE_VARIANT_IMAGENET50_96)
#include "squeezenet1_0_imagenet50_96_relu_on_maxpool_off.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet50_96_relu_on_maxpool_off
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_OFF_output
#define BUNDLE_OUTPUT_ELEMENTS 50
#elif defined(BUNDLE_CASE_RELU_ON_MAXPOOL_ON) && defined(BUNDLE_VARIANT_IMAGENET50_96)
#include "squeezenet1_0_imagenet50_96_relu_on_maxpool_on.h"
#define BUNDLE_ENTRY squeezenet1_0_imagenet50_96_relu_on_maxpool_on
#define BUNDLE_CONSTANT_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_CONSTANT_MEM_SIZE
#define BUNDLE_MUTABLE_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_MUTABLE_MEM_SIZE
#define BUNDLE_ACTIVATIONS_MEM_SIZE SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_ACTIVATIONS_MEM_SIZE
#define BUNDLE_MEM_ALIGN SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_MEM_ALIGN
#define BUNDLE_DATA_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_data
#define BUNDLE_OUTPUT_OFFSET SQUEEZENET1_0_IMAGENET50_96_RELU_ON_MAXPOOL_ON_output
#define BUNDLE_OUTPUT_ELEMENTS 50
#else
#error "Define one bundle case and one bundle variant macro."
#endif

namespace {

constexpr size_t kInputN = 1;
#if defined(BUNDLE_VARIANT_IMAGENET100_224)
constexpr size_t kInputC = 3;
constexpr size_t kInputH = 224;
constexpr size_t kInputW = 224;
#elif defined(BUNDLE_VARIANT_IMAGENET32)
constexpr size_t kInputC = 3;
constexpr size_t kInputH = 128;
constexpr size_t kInputW = 128;
#elif defined(BUNDLE_VARIANT_CIFAR32)
constexpr size_t kInputC = 3;
constexpr size_t kInputH = 32;
constexpr size_t kInputW = 32;
#else
constexpr size_t kInputC = 3;
constexpr size_t kInputH = 96;
constexpr size_t kInputW = 96;
#endif
constexpr size_t kInputBytes =
    kInputN * kInputC * kInputH * kInputW * sizeof(float);

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
  for (size_t index = 0; index < count; ++index) {
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

int runStreamMode(uint8_t *constantWeight, uint8_t *mutableWeight,
                  uint8_t *activations) {
  std::vector<char> inputBuffer(kInputBytes);
  const size_t outputBytes = BUNDLE_OUTPUT_ELEMENTS * sizeof(float);

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
      argc >= 4 ? argv[3] : std::string("bundle_output.txt");
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
  writeTextOutputOrExit(outputPath, output, BUNDLE_OUTPUT_ELEMENTS);
  writeBinaryOutputOrExit(outputBinPath, output, BUNDLE_OUTPUT_ELEMENTS);

  return 0;
}
