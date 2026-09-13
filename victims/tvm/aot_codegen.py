"""AOT kernel plan + generated runner (from paper TVM matrix, artifact-local)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tvm_backend import load_tvm_modules

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.shape))


@dataclass(frozen=True)
class ConstantEntry:
    var_name: str
    filename: str
    spec: TensorSpec


@dataclass(frozen=True)
class InputRef:
    kind: str
    name: str


@dataclass(frozen=True)
class CallEntry:
    output_name: str
    kernel_name: str
    inputs: tuple[InputRef, ...]
    output_spec: TensorSpec

def module_to_text(mod: Any, *, show_meta: bool = False) -> str:
    if hasattr(mod, "script"):
        try:
            return mod.script(show_meta=show_meta)
        except TypeError:
            return mod.script()
    if hasattr(mod, "astext"):
        return mod.astext()
    return str(mod)


def tensor_spec_from_sinfo(sinfo: Any) -> TensorSpec:
    shape_expr = getattr(sinfo, "shape", None)
    if shape_expr is None or not hasattr(shape_expr, "values"):
        raise TypeError(f"expected static tensor shape, got {type(sinfo)}")
    shape = tuple(int(dim) for dim in shape_expr.values)
    dtype = str(getattr(sinfo, "dtype", ""))
    if not dtype:
        raise TypeError(f"missing dtype on tensor struct info: {sinfo}")
    return TensorSpec(shape=shape, dtype=dtype)


def expr_key(expr: Any) -> Any:
    vid = getattr(expr, "vid", None)
    if vid is not None:
        return vid
    return expr


def extract_kernel_module(lowered_mod: Any, tvm: Any) -> tuple[Any, list[str]]:
    kernel_funcs: dict[Any, Any] = {}
    kernel_names: list[str] = []
    for gvar, func in lowered_mod.functions.items():
        if isinstance(func, tvm.tirx.function.PrimFunc):
            kernel_name = gvar.name_hint
            kernel_funcs[gvar] = func.with_attr("global_symbol", kernel_name)
            kernel_names.append(kernel_name)
    if not kernel_funcs:
        raise RuntimeError("no TIR kernels found in lowered module")
    return tvm.IRModule(kernel_funcs, attrs=lowered_mod.attrs), kernel_names


def collect_lowered_call_sequence(
    lowered_mod: Any,
) -> tuple[Any, TensorSpec, Any, list[tuple[Any, str, TensorSpec, tuple[Any, ...]]]]:
    (
        _,
        _,
        _,
        Call,
        _,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    ) = load_tvm_modules()

    main = lowered_mod["main"]
    if len(main.params) != 1:
        raise ValueError(f"expected exactly one input parameter, got {len(main.params)}")
    input_var = main.params[0]
    input_spec = tensor_spec_from_sinfo(input_var.struct_info)

    aliases: dict[Any, Any] = {}
    resolve_cache: dict[Any, Any] = {}
    calls: list[tuple[Any, str, TensorSpec, tuple[Any, ...]]] = []

    def resolve_expr(expr: Any, resolving: set[int] | None = None) -> Any:
        if isinstance(expr, Tuple):
            return tuple(resolve_expr(field, resolving) for field in expr.fields)
        if isinstance(expr, TupleGetItem):
            tuple_value = resolve_expr(expr.tuple_value, resolving)
            if not isinstance(tuple_value, tuple):
                raise TypeError(f"expected tuple source for TupleGetItem, got {type(tuple_value)}")
            return resolve_expr(tuple_value[int(expr.index)], resolving)
        if isinstance(expr, (Var, DataflowVar)):
            key = expr_key(expr)
            if key in resolve_cache:
                return resolve_cache[key]
            target = aliases.get(key)
            if target is None:
                return expr
            if resolving is None:
                resolving = set()
            if key in resolving:
                return expr
            resolving.add(key)
            try:
                resolved = resolve_expr(target, resolving)
            finally:
                resolving.remove(key)
            resolve_cache[key] = resolved
            return resolved
        return expr

    for block in main.body.blocks:
        for binding in block.bindings:
            if not isinstance(binding, VarBinding):
                raise TypeError(f"unsupported binding type: {type(binding)}")
            value = binding.value
            if isinstance(value, Tuple):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if isinstance(value, TupleGetItem):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if isinstance(value, (Var, DataflowVar)):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if not isinstance(value, Call):
                raise TypeError(f"unsupported binding value: {type(value)}")
            if str(value.op) != "Op(relax.call_tir)":
                raise TypeError(f"unsupported call op: {value.op}")
            if len(value.args) != 2:
                raise ValueError("call_tir is expected to have two arguments")

            kernel_gvar = value.args[0]
            args_tuple = value.args[1]
            if not isinstance(kernel_gvar, GlobalVar):
                raise TypeError(f"unexpected call_tir target: {type(kernel_gvar)}")
            if not isinstance(args_tuple, Tuple):
                raise TypeError(f"unexpected call_tir args tuple: {type(args_tuple)}")

            calls.append(
                (
                    binding.var,
                    kernel_gvar.name_hint,
                    tensor_spec_from_sinfo(value.sinfo_args[0]),
                    tuple(resolve_expr(arg) for arg in args_tuple.fields),
                )
            )

    output_expr = resolve_expr(main.body.body)
    return input_var, input_spec, output_expr, calls


def collect_runner_plan(
    lowered_mod: Any,
) -> tuple[str, TensorSpec, str, TensorSpec, list[ConstantEntry], list[CallEntry]]:
    _, _, _, _, Constant, DataflowVar, _, _, _, Var, _ = load_tvm_modules()
    input_var, input_spec, output_expr, resolved_calls = collect_lowered_call_sequence(lowered_mod)
    input_name = input_var.name_hint
    if not isinstance(output_expr, (Var, DataflowVar)):
        raise TypeError(f"unsupported main output expression: {type(output_expr)}")

    constants: list[ConstantEntry] = []
    calls: list[CallEntry] = []
    const_index = 0
    used_names = {input_name}
    canonical_names: dict[Any, str] = {expr_key(input_var): input_name}

    def canonical_var_name(var: Any) -> str:
        key = expr_key(var)
        existing = canonical_names.get(key)
        if existing is not None:
            return existing
        base = getattr(var, "name_hint", "") or "tensor"
        name = base
        suffix = 1
        while name in used_names:
            name = f"{base}_{suffix:03d}"
            suffix += 1
        canonical_names[key] = name
        used_names.add(name)
        return name

    for output_var, kernel_name, output_spec, resolved_args in resolved_calls:
        input_refs: list[InputRef] = []
        for arg in resolved_args:
            if isinstance(arg, (Var, DataflowVar)):
                input_refs.append(InputRef(kind="tensor", name=canonical_var_name(arg)))
                continue
            if isinstance(arg, Constant):
                array = arg.data.numpy()
                if str(array.dtype) != "float32":
                    raise TypeError(f"unsupported constant dtype: {array.dtype}")
                const_name = f"const_{const_index:02d}"
                filename = f"{const_index:02d}_{const_name}.bin"
                constants.append(
                    ConstantEntry(
                        var_name=const_name,
                        filename=filename,
                        spec=TensorSpec(
                            shape=tuple(int(dim) for dim in array.shape),
                            dtype=str(array.dtype),
                        ),
                    )
                )
                input_refs.append(InputRef(kind="constant", name=const_name))
                const_index += 1
                continue
            raise TypeError(f"unsupported call_tir argument: {type(arg)}")

        calls.append(
            CallEntry(
                output_name=canonical_var_name(output_var),
                kernel_name=kernel_name,
                inputs=tuple(input_refs),
                output_spec=output_spec,
            )
        )

    output_name = canonical_var_name(output_expr)
    output_spec = tensor_spec_from_sinfo(output_expr.struct_info)
    return input_name, input_spec, output_name, output_spec, constants, calls


def export_constant_blobs(*, lowered_mod: Any, constants_dir: Path) -> list[dict[str, Any]]:
    _, _, _, _, Constant, _, _, _, _, _, _ = load_tvm_modules()
    constants_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, resolved_calls = collect_lowered_call_sequence(lowered_mod)
    entries: list[dict[str, Any]] = []
    const_index = 0
    for _, _, _, resolved_args in resolved_calls:
        for arg in resolved_args:
            if not isinstance(arg, Constant):
                continue
            array = np.asarray(arg.data.numpy(), dtype=np.float32)
            const_name = f"const_{const_index:02d}"
            filename = f"{const_index:02d}_{const_name}.bin"
            path = constants_dir / filename
            array.tofile(path)
            entries.append(
                {
                    "index": int(const_index),
                    "var_name": const_name,
                    "filename": filename,
                    "shape": [int(dim) for dim in array.shape],
                    "dtype": str(array.dtype),
                    "num_elements": int(array.size),
                    "nbytes": int(array.nbytes),
                }
            )
            const_index += 1
    return entries


def shape_literal(shape: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(int(dim)) for dim in shape) + "}"


def shape_json(shape: tuple[int, ...]) -> str:
    return ", ".join(str(int(dim)) for dim in shape)


def generate_runner_source(
    *,
    input_name: str,
    input_spec: TensorSpec,
    output_name: str,
    output_spec: TensorSpec,
    constants: list[ConstantEntry],
    calls: list[CallEntry],
    library_filename: str,
) -> str:
    kernels: list[str] = []
    for call in calls:
        if call.kernel_name not in kernels:
            kernels.append(call.kernel_name)

    const_lines = []
    for const in constants:
        const_lines.append(
            "    Tensor {name} = TensorFromFile(JoinPath(args.constants_dir, {filename}), {elems}, {shape});".format(
                name=const.var_name,
                filename=json.dumps(const.filename),
                elems=const.spec.num_elements,
                shape=shape_literal(const.spec.shape),
            )
        )

    tensor_alloc_lines = []
    call_lines = []
    tensor_storage_names: dict[str, str] = {input_name: input_name}
    for index, call in enumerate(calls):
        storage_name = f"tensor_{index:04d}"
        tensor_storage_names[call.output_name] = storage_name
        tensor_alloc_lines.append(
            f"    Tensor {storage_name} = EmptyTensor({shape_literal(call.output_spec.shape)});"
        )

        arg_names: list[str] = []
        for ref in call.inputs:
            if ref.kind == "constant":
                arg_names.append(ref.name)
                continue
            if ref.kind != "tensor":
                raise ValueError(f"unsupported input ref kind: {ref.kind}")
            storage_ref = tensor_storage_names.get(ref.name)
            if storage_ref is None:
                raise KeyError(f"missing tensor storage for {ref.name}")
            arg_names.append(storage_ref)
        call_lines.append(f"    {call.kernel_name}({', '.join(arg_names)}, {storage_name});")

    output_storage_name = tensor_storage_names.get(output_name, output_name)

    kernel_decl_lines = [
        f'    Function {kernel} = RequireFunction(kernels, "{kernel}");' for kernel in kernels
    ]

    return f"""#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/input_zero_dither.h>
#include <tvm/runtime/relu_low12_patch.h>
#include <tvm/runtime/tensor.h>

namespace {{

using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::runtime::DataType;
using tvm::runtime::Tensor;

constexpr int64_t kInputElems = {input_spec.num_elements};
constexpr int64_t kOutputElems = {output_spec.num_elements};

struct Args {{
  std::string library_path;
  std::string constants_dir;
  std::string input_bin;
  std::string output_bin;
  std::string output_txt;
  std::string summary_json;
  bool stream_mode{{false}};
}};

void Usage(const char* argv0) {{
  std::cerr << "Usage: " << argv0
            << " --library <{library_filename}>"
            << " --constants-dir <constants_dir>"
            << " [--input-bin <input_f32.bin>]"
            << " [--output-bin <output_f32.bin>]"
            << " [--output-txt <output.txt>]"
            << " [--summary-json <summary.json>]"
            << " [--stream]\\n";
}}

std::string RequireValue(int& i, int argc, char** argv, const char* flag) {{
  if (i + 1 >= argc) {{
    throw std::runtime_error(std::string("missing value for ") + flag);
  }}
  ++i;
  return argv[i];
}}

Args ParseArgs(int argc, char** argv) {{
  Args out;
  for (int i = 1; i < argc; ++i) {{
    std::string arg = argv[i];
    if (arg == "--library") {{
      out.library_path = RequireValue(i, argc, argv, "--library");
    }} else if (arg == "--constants-dir") {{
      out.constants_dir = RequireValue(i, argc, argv, "--constants-dir");
    }} else if (arg == "--input-bin") {{
      out.input_bin = RequireValue(i, argc, argv, "--input-bin");
    }} else if (arg == "--output-bin") {{
      out.output_bin = RequireValue(i, argc, argv, "--output-bin");
    }} else if (arg == "--output-txt") {{
      out.output_txt = RequireValue(i, argc, argv, "--output-txt");
    }} else if (arg == "--summary-json") {{
      out.summary_json = RequireValue(i, argc, argv, "--summary-json");
    }} else if (arg == "--stream") {{
      out.stream_mode = true;
    }} else if (arg == "-h" || arg == "--help") {{
      Usage(argv[0]);
      std::exit(0);
    }} else {{
      throw std::runtime_error("unknown argument: " + arg);
    }}
  }}

  if (out.library_path.empty() || out.constants_dir.empty()) {{
    throw std::runtime_error("missing required arguments");
  }}
  if (!out.stream_mode && (out.input_bin.empty() || out.output_bin.empty())) {{
    throw std::runtime_error("missing required arguments");
  }}
  return out;
}}

std::vector<float> ReadFloat32File(const std::string& path, int64_t expected_elems) {{
  std::ifstream in(path, std::ios::binary);
  if (!in) {{
    throw std::runtime_error("failed to open input file: " + path);
  }}
  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size < 0) {{
    throw std::runtime_error("failed to stat input file: " + path);
  }}
  const auto expected_bytes =
      expected_elems * static_cast<int64_t>(sizeof(float));
  if (size != static_cast<std::streamoff>(expected_bytes)) {{
    std::ostringstream os;
    os << "unexpected input size for " << path << ": got " << size << " bytes, expected "
       << expected_bytes << " bytes";
    throw std::runtime_error(os.str());
  }}
  std::vector<float> data(static_cast<size_t>(expected_elems));
  in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in) {{
    throw std::runtime_error("failed to read input file: " + path);
  }}
  return data;
}}

bool ReadExact(std::istream& in, void* dst, size_t nbytes) {{
  char* out = static_cast<char*>(dst);
  size_t total = 0;
  while (total < nbytes) {{
    in.read(out + total, static_cast<std::streamsize>(nbytes - total));
    const size_t got = static_cast<size_t>(in.gcount());
    if (got == 0) {{
      if (total == 0 && in.eof()) {{
        return false;
      }}
      throw std::runtime_error("unexpected EOF while reading streamed input");
    }}
    total += got;
  }}
  return true;
}}

std::string JoinPath(const std::string& lhs, const std::string& rhs) {{
  if (lhs.empty()) {{
    return rhs;
  }}
  if (lhs.back() == '/') {{
    return lhs + rhs;
  }}
  return lhs + "/" + rhs;
}}

void WriteFloat32File(const std::string& path, const std::vector<float>& data) {{
  std::ofstream out(path, std::ios::binary);
  if (!out) {{
    throw std::runtime_error("failed to open output bin: " + path);
  }}
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!out) {{
    throw std::runtime_error("failed to write output bin: " + path);
  }}
}}

void WriteTextFile(const std::string& path, const std::vector<float>& data) {{
  std::ofstream out(path);
  if (!out) {{
    throw std::runtime_error("failed to open output txt: " + path);
  }}
  out << std::setprecision(9);
  for (size_t i = 0; i < data.size(); ++i) {{
    out << i << '\\t' << data[i] << '\\n';
  }}
  if (!out) {{
    throw std::runtime_error("failed to write output txt: " + path);
  }}
}}

std::vector<int> TopKIndices(const std::vector<float>& data, size_t k) {{
  std::vector<int> indices(data.size());
  for (size_t i = 0; i < data.size(); ++i) {{
    indices[i] = static_cast<int>(i);
  }}
  if (k > indices.size()) {{
    k = indices.size();
  }}
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                    indices.end(), [&data](int lhs, int rhs) {{
                      return data[static_cast<size_t>(lhs)] >
                             data[static_cast<size_t>(rhs)];
                    }});
  indices.resize(k);
  return indices;
}}

void WriteSummaryJson(const std::string& path, const Args& args, double tvm_seconds,
                      const std::vector<float>& output) {{
  std::ofstream out(path);
  if (!out) {{
    throw std::runtime_error("failed to open summary json: " + path);
  }}
  const auto top3 = TopKIndices(output, 3);
  const int pred = static_cast<int>(
      std::distance(output.begin(), std::max_element(output.begin(), output.end())));

  out << "{{\\n";
  out << "  \\"input_bin\\": " << std::quoted(args.input_bin) << ",\\n";
  out << "  \\"library_path\\": " << std::quoted(args.library_path) << ",\\n";
  out << "  \\"constants_dir\\": " << std::quoted(args.constants_dir) << ",\\n";
  out << "  \\"tvm_seconds\\": " << std::fixed << std::setprecision(6) << tvm_seconds << ",\\n";
  out << "  \\"tvm\\": {{\\n";
  out << "    \\"name\\": \\"tvm_aot\\",\\n";
  out << "    \\"shape\\": [{shape_json(output_spec.shape)}],\\n";
  out << "    \\"pred\\": " << pred << ",\\n";
  out << "    \\"top3_indices\\": [";
  for (size_t i = 0; i < top3.size(); ++i) {{
    out << top3[i];
    if (i + 1 < top3.size()) {{
      out << ", ";
    }}
  }}
  out << "],\\n";
  out << "    \\"top3_logits\\": [";
  for (size_t i = 0; i < top3.size(); ++i) {{
    out << output[static_cast<size_t>(top3[i])];
    if (i + 1 < top3.size()) {{
      out << ", ";
    }}
  }}
  out << "],\\n";
  float sum = 0.0f;
  float min_v = output.front();
  float max_v = output.front();
  for (float value : output) {{
    sum += value;
    min_v = std::min(min_v, value);
    max_v = std::max(max_v, value);
  }}
  out << "    \\"logits_sum\\": " << sum << ",\\n";
  out << "    \\"logits_min\\": " << min_v << ",\\n";
  out << "    \\"logits_max\\": " << max_v << "\\n";
  out << "  }}\\n";
  out << "}}\\n";
  if (!out) {{
    throw std::runtime_error("failed to write summary json: " + path);
  }}
}}

Function RequireFunction(const Module& mod, const std::string& name) {{
  auto opt = mod->GetFunction(name, false);
  if (!opt.has_value()) {{
    throw std::runtime_error("missing module function: " + name);
  }}
  return *opt;
}}

Tensor EmptyTensor(std::initializer_list<int64_t> shape) {{
  return Tensor::Empty(tvm::ffi::Shape(shape), DataType(DataType::kFloat, 32, 1),
                       DLDevice{{kDLCPU, 0}});
}}

Tensor TensorFromFile(const std::string& path, int64_t expected_elems,
                      std::initializer_list<int64_t> shape) {{
  const std::vector<float> data = ReadFloat32File(path, expected_elems);
  Tensor tensor = EmptyTensor(shape);
  tensor.CopyFromBytes(data.data(), data.size() * sizeof(float));
  return tensor;
}}

Tensor InputTensorFromFile(const std::string& path, int64_t expected_elems,
                           std::initializer_list<int64_t> shape) {{
  const std::vector<float> data = ReadFloat32File(path, expected_elems);
  Tensor tensor = EmptyTensor(shape);
  const size_t nbytes = data.size() * sizeof(float);
  if (!tvm::runtime::inputzerodither::TryCopyFromBytes(
          data.data(), nbytes, const_cast<DLTensor*>(tensor.operator->()))) {{
    tensor.CopyFromBytes(data.data(), nbytes);
  }}
  return tensor;
}}

Tensor InputTensorFromBytes(const float* data, int64_t expected_elems,
                            std::initializer_list<int64_t> shape) {{
  Tensor tensor = EmptyTensor(shape);
  const size_t nbytes =
      static_cast<size_t>(expected_elems) * sizeof(float);
  if (!tvm::runtime::inputzerodither::TryCopyFromBytes(
          data, nbytes, const_cast<DLTensor*>(tensor.operator->()))) {{
    tensor.CopyFromBytes(data, nbytes);
  }}
  return tensor;
}}

}}  // namespace

int main(int argc, char** argv) {{
  try {{
    const Args args = ParseArgs(argc, argv);

    Module kernels = Module::LoadFromFile(args.library_path);
{chr(10).join(kernel_decl_lines)}
{chr(10).join(const_lines)}
{chr(10).join(tensor_alloc_lines)}
    if (args.stream_mode) {{
      std::vector<float> input(static_cast<size_t>(kInputElems));
      std::vector<float> output(static_cast<size_t>(kOutputElems));
      while (ReadExact(std::cin, input.data(), input.size() * sizeof(float))) {{
        Tensor {input_name} =
            InputTensorFromBytes(input.data(), kInputElems, {shape_literal(input_spec.shape)});
        tvm::runtime::relulow12::ResetInferencePatchSeed();
{chr(10).join(call_lines)}
        {output_storage_name}.CopyToBytes(output.data(), output.size() * sizeof(float));
        std::cout.write(reinterpret_cast<const char*>(output.data()),
                        static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!std::cout) {{
          throw std::runtime_error("failed to write streamed output");
        }}
        std::cout.flush();
      }}
      return 0;
    }}

    Tensor {input_name} =
        InputTensorFromFile(args.input_bin, kInputElems, {shape_literal(input_spec.shape)});
    const auto start = std::chrono::steady_clock::now();
    tvm::runtime::relulow12::ResetInferencePatchSeed();
{chr(10).join(call_lines)}
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::vector<float> output(static_cast<size_t>(kOutputElems));
    {output_storage_name}.CopyToBytes(output.data(), output.size() * sizeof(float));

    WriteFloat32File(args.output_bin, output);
    if (!args.output_txt.empty()) {{
      WriteTextFile(args.output_txt, output);
    }}
    if (!args.summary_json.empty()) {{
      WriteSummaryJson(args.summary_json, args, elapsed.count(), output);
    }}
    return 0;
  }} catch (const std::exception& ex) {{
    std::cerr << "error: " << ex.what() << '\\n';
    return 1;
  }}
}}
"""

