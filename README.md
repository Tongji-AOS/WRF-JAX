# WRF-JAX

**English** | [简体中文](README.zh-CN.md)

A Python/JAX implementation of the WRF 4.5 `em_real` configuration, including the dynamical core and Thompson, Tiedtke, MYJ, Noah, and RRTMG physics schemes. Supports CPUs and NVIDIA GPUs.

![WRF-JAX animation](infa.gif)

Run the following commands from the project root, choosing the environment for your device.

GPU environment:

```bash
conda env create -n wrf-jax-gpu -f environment-gpu.yml
conda activate wrf-jax-gpu
export JAX_PLATFORMS=cuda
```

CPU environment:

```bash
conda env create -n wrf-jax -f environment.yml
conda activate wrf-jax
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_parallel_codegen_split_count=16 --xla_allow_excess_precision=false"
```

Prepare a WRF 4.5 `em_real` single-domain case directory containing these inputs and parameter tables:

| Contents | Files |
|---|---|
| Case configuration, initial conditions, and boundary conditions | `namelist.input`, `wrfinput_d01`, `wrfbdy_d01` |
| Land surface parameter tables | `LANDUSE.TBL`, `VEGPARM.TBL`, `SOILPARM.TBL`, `GENPARM.TBL` |
| Thompson lookup tables | `qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`, `freezeH2O.dat`, `CCN_ACTIVATE.BIN` |
| Ozone data | `ozone.formatted`, `ozone_lat.formatted`, `ozone_plev.formatted` |
| Greenhouse gas data (`ghg_input=1`) | `CAMtr_volume_mixing_ratio` |

Bundled radiation and surface exchange lookup tables are in `src/wrf_jax/physics/data/`.

Run a one-hour forecast using Python from the active Conda environment. Replace `/path/to/case` with your case directory.

```bash
export WRF_PY_BOOTSTRAP_PYTHON=0
export WRF_PY_COMPARE_STREAMED_SNAPSHOTS=0
export WRF_PY_KEEP_STREAMED_SNAPSHOTS=0

python main.py \
  --case-dir /path/to/case \
  --out-dir outputs/example \
  --duration-seconds 3600 \
  --snapshot-interval-seconds 3600 \
  --mode compiled-direct
```

The default configuration is `config/default_run.json`, with `compiled-direct` as the default execution mode. Set the forecast duration with `--duration-seconds` and the output interval with `--snapshot-interval-seconds`. Both values are in seconds and should be integer multiples of the case time step.

To compare against original WRF results, set `WRF_PY_COMPARE_STREAMED_SNAPSHOTS=1` and add `--wrfout /path/to/wrfout` to the run command. Use a reference file containing records for each output time.

Results are written to `outputs/example/`:

- `snapshots/elapsed-*/wrfjax_uvtq_steps.nc`: model fields at each output time.
- `run.log`, `run_summary.txt`: run log and summary.
- `timing_summary.md`: integration timing statistics.
- When comparison is enabled, each snapshot directory also contains `comparison.md` and `nrmse_summary.txt`.
