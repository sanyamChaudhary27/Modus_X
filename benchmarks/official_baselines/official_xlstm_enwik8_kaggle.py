import base64

SCRIPT = r"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path("/kaggle/working/official_xlstm_enwik8_80m_meshdp_denseaudit")
ROOT.mkdir(parents=True, exist_ok=True)
REPO = ROOT / "xlstm-jax"
OUT = ROOT / "results_meshdp"
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "dacite", "mlstm-kernels"],
    check=True,
)
for module_name in list(sys.modules):
    if module_name == "datasets" or module_name.startswith("datasets."):
        del sys.modules[module_name]
    if module_name == "pyarrow" or module_name.startswith("pyarrow."):
        del sys.modules[module_name]
import pyarrow as pa
if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType
print("XLSTM80_MESHDP_DEPS_READY", flush=True)

if not REPO.exists():
    print("XLSTM80_MESHDP_CLONE_XLSTM_JAX", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NX-AI/xlstm-jax.git", str(REPO)], check=True)

sys.path.insert(0, str(REPO))

print("XLSTM80_MESHDP_IMPORTS_START", flush=True)
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import mLSTMBackendNameAndKwargs
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.training import get_num_params, get_train_step_fn, init_xlstm
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

devices = jax.devices()
print("XLSTM80_MESHDP_JAX_DEVICES", [str(d) for d in devices], flush=True)
if len(devices) < 8:
    raise RuntimeError(f"Expected Kaggle TPU with 8 devices, got {len(devices)}")

DATA = ROOT / "enwik8"
if not DATA.exists():
    z = ROOT / "enwik8.zip"
    print("XLSTM80_MESHDP_DOWNLOAD_ENWIK8", flush=True)
    urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", z)
    with zipfile.ZipFile(z) as archive:
        archive.extractall(ROOT)

raw = np.frombuffer(DATA.read_bytes(), dtype=np.uint8)
train = raw[:90_000_000]
valid = raw[90_000_000:95_000_000]
test = raw[95_000_000:100_000_000]


def make_parallel_config():
    return ParallelConfig(
        data_axis_size=-1,
        fsdp_axis_size=1,
        pipeline_axis_size=1,
        model_axis_size=1,
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        pipeline_axis_name="pp",
        model_axis_name="tp",
        fsdp_modules=(),
        fsdp_min_weight_size=2**18,
        remat=(),
    )


def make_model_config(parallel, seq_len=512, embed_dim=1024, n_layers=12, n_heads=8, dtype="bfloat16"):
    backend = mLSTMBackendNameAndKwargs(name="parallel_stabilized", kwargs={})
    cell = mLSTMCellConfig(
        context_length=seq_len,
        embedding_dim=embed_dim,
        num_heads=n_heads,
        backend=backend,
        dtype=dtype,
        gate_dtype="float32",
        gate_linear_headwise=True,
        parallel=parallel,
    )
    layer = mLSTMLayerConfig(
        embedding_dim=embed_dim,
        context_length=seq_len,
        num_heads=n_heads,
        dtype=dtype,
        mlstm_cell=cell,
        parallel=parallel,
    )
    block = mLSTMBlockConfig(mlstm=layer, parallel=parallel)
    return xLSTMLMModelConfig(
        vocab_size=256,
        context_length=seq_len,
        num_blocks=n_layers,
        embedding_dim=embed_dim,
        mlstm_block=block,
        parallel=parallel,
        dtype=dtype,
        tie_weights=False,
        dropout=0.0,
    )


def make_batch(data, step, *, global_batch=8, seq_len=512):
    x = np.empty((global_batch, seq_len), dtype=np.int32)
    y = np.empty((global_batch, seq_len), dtype=np.int32)
    stride = seq_len + 1
    for b in range(global_batch):
        pos = ((step * global_batch + b) * stride) % (len(data) - stride)
        chunk = data[pos : pos + stride].astype(np.int32)
        x[b] = chunk[:-1]
        y[b] = chunk[1:]
    return LLMBatch.from_inputs(jnp.asarray(x), targets=jnp.asarray(y))


def metrics_loss_bpc(metrics):
    loss_sum, count = metrics["loss"]
    return float(loss_sum / count / jnp.log(2.0))


def run_case(name, *, seq_len, embed_dim, n_layers, n_heads, steps, checkpoint_step, lr, smoke=False):
    parallel = make_parallel_config()
    mesh = initialize_mesh(parallel_config=parallel)
    cfg = make_model_config(parallel, seq_len=seq_len, embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads)
    chars_per_step = 8 * seq_len
    case_config = {
        "seq_len": seq_len,
        "global_batch": 8,
        "chars_per_step": chars_per_step,
        "embed_dim": embed_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "lr": lr,
        "parallel": {
            "data_axis_size": parallel.data_axis_size,
            "fsdp_axis_size": parallel.fsdp_axis_size,
            "pipeline_axis_size": parallel.pipeline_axis_size,
            "model_axis_size": parallel.model_axis_size,
            "axis_names": mesh.axis_names,
            "mesh_shape": str(mesh.devices.shape),
        },
        "note": "Official NXAI xlstm-jax mLSTM baseline using repo-native shard_map data parallelism: dp=8, tp=1.",
    }
    print("XLSTM80_MESHDP_CASE_START", name, json.dumps(case_config), flush=True)

    optimizer = optax.adamw(lr, weight_decay=0.1)
    init_batch = make_batch(train, 0, seq_len=seq_len)
    state = init_xlstm(
        config=cfg,
        mesh=mesh,
        rng=jax.random.PRNGKey(1),
        input_array=init_batch.inputs,
        optimizer=optimizer,
    )
    param_count = int(get_num_params(state))
    print("XLSTM80_MESHDP_PARAMS", name, param_count, flush=True)
    if not (70_000_000 <= param_count <= 90_000_000):
        raise RuntimeError(f"Expected an 80M-tier xLSTM baseline, got {param_count:,} params")

    train_step, train_metrics = get_train_step_fn(state, init_batch, mesh, parallel)

    model = xLSTMLMModel(cfg)

    def eval_loss_fn(state, batch, metrics):
        logits = state.apply_fn({"params": state.params}, batch.inputs, train=False)
        labels = batch.targets
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
        step_metrics = {"loss": (loss.sum(), loss.size)}
        step_metrics = jax.tree.map(
            lambda x: jax.lax.psum(x, axis_name=(parallel.data_axis_name, parallel.fsdp_axis_name, parallel.pipeline_axis_name, parallel.model_axis_name)),
            step_metrics,
        )
        if metrics is None:
            return step_metrics
        return jax.tree.map(jnp.add, metrics, step_metrics)

    state_specs = nn.get_partition_spec(state)
    eval_step = jax.jit(
        shard_map(
            eval_loss_fn,
            mesh,
            in_specs=(state_specs, P((parallel.data_axis_name, parallel.fsdp_axis_name)), P()),
            out_specs=P(),
            check_rep=False,
        )
    )

    def eval_bpc(data, chunks=64):
        metrics = None
        for i in range(chunks):
            metrics = eval_step(state, make_batch(data, i, seq_len=seq_len), metrics)
        return metrics_loss_bpc(jax.device_get(metrics))

    def make_position_batch(data, positions):
        x = np.empty((8, seq_len), dtype=np.int32)
        y = np.empty((8, seq_len), dtype=np.int32)
        if len(positions) == 0:
            raise ValueError("positions must be non-empty")
        padded = list(positions)
        while len(padded) < 8:
            padded.append(padded[-1])
        for b, pos in enumerate(padded[:8]):
            chunk = data[pos : pos + seq_len + 1].astype(np.int32)
            x[b] = chunk[:-1]
            y[b] = chunk[1:]
        return LLMBatch.from_inputs(jnp.asarray(x), targets=jnp.asarray(y))

    def eval_position_bpc(data, positions):
        metrics = None
        for start in range(0, len(positions), 8):
            metrics = eval_step(state, make_position_batch(data, positions[start : start + 8]), metrics)
        return metrics_loss_bpc(jax.device_get(metrics))

    def audit_split(label, data):
        max_start = len(data) - seq_len - 1
        rng = np.random.default_rng(12345)
        linspace_positions = np.linspace(0, max_start, 1024, dtype=np.int64).tolist()
        random_positions = np.sort(rng.integers(0, max_start + 1, size=1024, dtype=np.int64)).tolist()
        dense0 = list(range(0, max_start + 1, seq_len))
        dense_half = list(range(seq_len // 2, max_start + 1, seq_len))
        rows = {
            "linspace": {"windows": len(linspace_positions), "bpc": eval_position_bpc(data, linspace_positions)},
            "random": {"windows": len(random_positions), "bpc": eval_position_bpc(data, random_positions)},
            "dense_offset_0": {"windows": len(dense0), "bpc": eval_position_bpc(data, dense0)},
            "dense_offset_half": {"windows": len(dense_half), "bpc": eval_position_bpc(data, dense_half)},
        }
        for mode, row in rows.items():
            print("XLSTM80_MESHDP_AUDIT", label, mode, json.dumps(row), flush=True)
        return rows

    t0 = time.time()
    last_loss = None
    for step in range(1, steps + 1):
        batch = make_batch(train, step, seq_len=seq_len)
        state, train_metrics = train_step(state, train_metrics, batch)
        if step == 1 or step % 10 == 0:
            host_metrics = jax.device_get(train_metrics)
            last_loss = metrics_loss_bpc(host_metrics)
            chars_s = step * chars_per_step / max(time.time() - t0, 1e-6)
            print(f"XLSTM80_MESHDP_PROGRESS {name} step={step}/{steps} loss_bpc={last_loss:.4f} chars_s={chars_s:.0f}", flush=True)
            train_metrics = jax.tree.map(lambda x: jnp.zeros_like(x), train_metrics)
        if step % checkpoint_step == 0 or step == steps:
            if last_loss is None:
                last_loss = metrics_loss_bpc(jax.device_get(train_metrics))
            val_bpc = eval_bpc(valid, chunks=8 if smoke else 64)
            row = {
                "case": name,
                "step": step,
                "processed_characters": step * chars_per_step,
                "loss_bpc": last_loss,
                "val_bpc": val_bpc,
                "params": param_count,
                "elapsed_s": time.time() - t0,
                "chars_per_step": chars_per_step,
            }
            print("XLSTM80_MESHDP_CHECKPOINT", json.dumps(row), flush=True)
            (OUT / f"{name}_progress.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    test_bpc = eval_bpc(test, chunks=8 if smoke else 64)
    print("XLSTM80_MESHDP_FINAL_TEST_BPC", name, f"{test_bpc:.4f}", flush=True)
    result = {"name": name, "params": param_count, "test_bpc": test_bpc, "chars_per_step": chars_per_step}
    if not smoke:
        audit = {
            "train_tail": audit_split("train_tail", train[-5_000_000:]),
            "validation": audit_split("validation", valid),
            "test": audit_split("test", test),
        }
        result["dense_audit"] = audit
        (OUT / f"{name}_dense_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("XLSTM80_MESHDP_DENSE_AUDIT_READY", OUT / f"{name}_dense_audit.json", flush=True)
    return result


results = []
results.append(
    run_case(
        "official_xlstm_80m_meshdp_163m_denseaudit",
        seq_len=512,
        embed_dim=1024,
        n_layers=12,
        n_heads=8,
        steps=40000,
        checkpoint_step=5000,
        lr=3e-4,
        smoke=False,
    )
)
(OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("XLSTM80_MESHDP_TPU_RUN_READY", OUT, flush=True)
"""

encoded = base64.b64encode(SCRIPT.encode("utf-8")).decode("ascii")
exec(base64.b64decode(encoded).decode("utf-8"))
