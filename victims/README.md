# victims

Shared ONNX is under `models/`. Each backend compiles into its own tree, same case IDs:

```
models/                 ONNX + MANIFEST.txt
                        collect inputs: data/inputs/<family>/<variant>/<split>/idx*
glow/                   Glow compile (MC / IC)
  compile.sh
  runner_src/           MC bundle runners
  libs/MCxx/{off,on}/   MCxx_runner + MCxx.weights.bin
  libs/ICxx/meta.json   image-classifier sidecar (ONNX still in models/)
tvm/                    TVM compile (TV / TA)
  compile.sh
  libs/TVxx/{off,on}/   VM .so
  libs/TAxx/{off,on}/   AOT kernels + generated runner
```

Host packages: `environment` at the artifact root.
MC01 / IC01 / TV01 / TA01 share `models/lenet/mnist64/`.
MC04 / IC04 / TV04 / TA04 share `models/SqueezeNet/cifar32/`.
MC10 / IC10 / TV10 / TA10 share `models/resnet/chest_single_channel/`.

```bash
bash victims/glow/compile.sh --runtime mc --case MC01
bash victims/glow/compile.sh --runtime ic --case IC01
bash victims/glow/compile.sh

bash victims/tvm/compile.sh --runtime tv --case TV01
bash victims/tvm/compile.sh --runtime ta --case TA01
bash victims/tvm/compile.sh
```

Pin collect writes
`trace_out/rho_trans/{mc,ic,tv,ta}/<id>/{off,on}/{train,test}/idx*/`.
`{train,test}/idx*` is the sample's dataset position (same as attack).
Backend is required (no default):

```bash
bash scripts/trace/collect_rho_trans.sh mc --case MC01
bash scripts/trace/collect_rho_trans.sh tv --case TV01
bash scripts/trace/collect_rho_trans.sh tvm
bash scripts/trace/collect_rho_trans.sh all
python3 scripts/compute_victim_rho_trans.py --backend mc
python3 scripts/compute_victim_acc.py --backend mc --case MC01
python3 scripts/compute_victim_overhead.py --backend mc --case MC01
python3 scripts/compute_victim_rho_static.py --case MC01
```

Old `glow/bundles/<family>/<variant>/bundle_*` paths are symlinks to `libs/MCxx/{off,on}`.
