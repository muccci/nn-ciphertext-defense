#ifndef GLOW_VGG_ONOFF_MULSAMPLE_MODEL_COMMON_FIXEDADDRESSARENA_H
#define GLOW_VGG_ONOFF_MULSAMPLE_MODEL_COMMON_FIXEDADDRESSARENA_H

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

namespace glow_vgg_onoff_mulsample {

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace detail {

inline bool iequals(const char *lhs, const char *rhs) {
  if (!lhs || !rhs) {
    return lhs == rhs;
  }
  while (*lhs && *rhs) {
    unsigned char lc = static_cast<unsigned char>(*lhs);
    unsigned char rc = static_cast<unsigned char>(*rhs);
    if (lc >= 'A' && lc <= 'Z') {
      lc = static_cast<unsigned char>(lc - 'A' + 'a');
    }
    if (rc >= 'A' && rc <= 'Z') {
      rc = static_cast<unsigned char>(rc - 'A' + 'a');
    }
    if (lc != rc) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

inline bool parseBoolEnvDefault(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return defaultValue;
  }
  if (iequals(value, "0") || iequals(value, "off") || iequals(value, "false") ||
      iequals(value, "no")) {
    return false;
  }
  if (iequals(value, "1") || iequals(value, "on") || iequals(value, "true") ||
      iequals(value, "yes")) {
    return true;
  }
  std::cerr << "invalid " << name << "=" << value << "\n";
  std::exit(2);
}

inline uintptr_t parseAddressEnvOrDefault(const char *name,
                                          uintptr_t defaultValue) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return defaultValue;
  }

  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 0);
  if (!end || end[0] != '\0') {
    std::cerr << "invalid " << name << "=" << value << "\n";
    std::exit(2);
  }
  if (parsed > std::numeric_limits<uintptr_t>::max()) {
    std::cerr << "out-of-range " << name << "=" << value << "\n";
    std::exit(2);
  }
  return static_cast<uintptr_t>(parsed);
}

inline size_t pageSizeOrDefault() {
  const long value = ::sysconf(_SC_PAGESIZE);
  return value > 0 ? static_cast<size_t>(value) : 4096u;
}

inline size_t alignUp(size_t value, size_t alignment) {
  if (alignment == 0u) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0u ? value : value + (alignment - remainder);
}

inline void *alignedAllocOrExit(size_t alignment, size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0 || !ptr) {
    std::cerr << "failed to allocate " << size << " bytes\n";
    std::exit(1);
  }
  std::memset(ptr, 0, size);
  return ptr;
}

} // namespace detail

class BundleMemoryArena {
public:
  BundleMemoryArena(size_t alignment, size_t constantSize, size_t mutableSize,
                    size_t activationsSize)
      : constantSize_(constantSize), mutableSize_(mutableSize),
        activationsSize_(activationsSize) {
    if (detail::parseBoolEnvDefault("GLOW_BUNDLE_FIXED_ARENA", true)) {
      allocateFixed(alignment);
    } else {
      allocateHeap(alignment);
    }
  }

  BundleMemoryArena(const BundleMemoryArena &) = delete;
  BundleMemoryArena &operator=(const BundleMemoryArena &) = delete;

  ~BundleMemoryArena() { release(); }

  uint8_t *constantWeight() const { return constantWeight_; }
  uint8_t *mutableWeight() const { return mutableWeight_; }
  uint8_t *activations() const { return activations_; }

private:
  static constexpr uintptr_t kDefaultFixedArenaBase = 0x400000000000ULL;

  void allocateFixed(size_t alignment) {
    const size_t pageSize = detail::pageSizeOrDefault();
    const size_t regionAlign = std::max(alignment, pageSize);
    const size_t constantOffset = 0u;
    const size_t mutableOffset =
        detail::alignUp(constantOffset + constantSize_, regionAlign);
    const size_t activationsOffset =
        detail::alignUp(mutableOffset + mutableSize_, regionAlign);
    const size_t totalSize =
        detail::alignUp(activationsOffset + activationsSize_, pageSize);

    const uintptr_t requestedBase = detail::parseAddressEnvOrDefault(
        "GLOW_BUNDLE_FIXED_ARENA_BASE", kDefaultFixedArenaBase);
    if (requestedBase % pageSize != 0u) {
      std::cerr << "GLOW_BUNDLE_FIXED_ARENA_BASE must be page aligned: 0x"
                << std::hex << requestedBase << std::dec << "\n";
      std::exit(2);
    }

    void *mapping = ::mmap(reinterpret_cast<void *>(requestedBase), totalSize,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                           -1, 0);
    if (mapping == MAP_FAILED) {
      const int savedErrno = errno;
      std::cerr << "failed to mmap fixed arena at 0x" << std::hex
                << requestedBase << std::dec << " size=" << totalSize
                << " errno=" << savedErrno << " (" << std::strerror(savedErrno)
                << ")\n";
      std::exit(1);
    }

    useFixedArena_ = true;
    mapping_ = mapping;
    mappingSize_ = totalSize;
    auto *base = static_cast<uint8_t *>(mapping);
    constantWeight_ = base + constantOffset;
    mutableWeight_ = base + mutableOffset;
    activations_ = base + activationsOffset;
  }

  void allocateHeap(size_t alignment) {
    constantWeight_ = static_cast<uint8_t *>(
        detail::alignedAllocOrExit(alignment, constantSize_));
    mutableWeight_ = static_cast<uint8_t *>(
        detail::alignedAllocOrExit(alignment, mutableSize_));
    activations_ = static_cast<uint8_t *>(
        detail::alignedAllocOrExit(alignment, activationsSize_));
  }

  void release() {
    if (useFixedArena_) {
      if (mapping_) {
        ::munmap(mapping_, mappingSize_);
      }
      mapping_ = nullptr;
      mappingSize_ = 0u;
    } else {
      std::free(constantWeight_);
      std::free(mutableWeight_);
      std::free(activations_);
    }

    constantWeight_ = nullptr;
    mutableWeight_ = nullptr;
    activations_ = nullptr;
  }

  bool useFixedArena_{false};
  void *mapping_{nullptr};
  size_t mappingSize_{0u};
  size_t constantSize_{0u};
  size_t mutableSize_{0u};
  size_t activationsSize_{0u};
  uint8_t *constantWeight_{nullptr};
  uint8_t *mutableWeight_{nullptr};
  uint8_t *activations_{nullptr};
};

} // namespace glow_vgg_onoff_mulsample

#endif
