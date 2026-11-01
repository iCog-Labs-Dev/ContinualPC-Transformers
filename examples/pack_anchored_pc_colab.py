"""Package explicit source/data inputs and a Colab notebook without Git pushes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def package(source: Path, assets: Path, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=source)
        .decode()
        .split("\0")
    )
    names = {
        n
        for n in tracked
        if n
        and (
            (n.startswith(("fabricpc/", "examples/", "tests/")) and n.endswith(".py"))
            or n in {"pyproject.toml", "README.md", "LICENSE", "jax_setup.py"}
        )
    }
    names.update({"fabricpc/experiments/anchored_pc.py", "tests/test_anchored_pc.py"})
    inputs = {f"src/{name}": source / name for name in sorted(names)}
    for name in (
        "checkpoints/bert_mini_128.json",
        "checkpoints/bert_mini_128.json.npz",
        "checkpoints/bert_mini_128.tokenizer.json",
        "data/domain_forgetting/news/manifest.json",
        "data/domain_forgetting/news/train.npy",
        "data/domain_forgetting/news/valid_windows.npz",
    ):
        inputs[name] = assets / name
    manifest = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in inputs.items()
        },
    }
    archive = output / "anchored_pc_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in inputs.items():
            tar.add(path, arcname=name, recursive=False)
        payload = json.dumps(manifest, indent=2).encode()
        member = tarfile.TarInfo("bundle_manifest.json")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()

    def markdown(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(True),
        }

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(True),
        }

    cells = [
        markdown(
            "# Anchored PC: Stage A diagnostic\n"
            "Fixed source, BERT-Mini checkpoint and news cache. No training loop.\n"
            "Use a free T4 GPU if available. The diagnostic stops at a failed anchor, "
            "failed convergence, or a two-hour budget. No Drive mount or GitHub push is needed.\n"
            f"Source base: `{manifest['base_revision']}`. Bundle SHA-256: `{archive_hash}`.\n"
        ),
        code(
            "from google.colab import files\n"
            "from pathlib import Path\n"
            "bundle = Path('/content/anchored_pc_bundle.tar.gz')\n"
            "if not bundle.exists():\n"
            "    uploaded = files.upload()\n"
            "    assert bundle.exists(), 'Upload anchored_pc_bundle.tar.gz'\n"
        ),
        code(
            "import hashlib, json, tarfile, subprocess, sys, os\n"
            f"assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '{archive_hash}'\n"
            "root = Path('/content/anchored_pc')\n"
            "root.mkdir(exist_ok=True)\n"
            "with tarfile.open(bundle) as archive:\n"
            "    archive.extractall(root, filter='data')\n"
            "manifest = json.loads((root / 'bundle_manifest.json').read_text())\n"
            "for name, expected in manifest['files'].items():\n"
            "    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name\n"
            "print('Verified', len(manifest['files']), 'source and input files')\n"
            "# Avoid shadowing by an unrelated installed tests package in Colab.\n"
            "(root / 'src/tests/__init__.py').touch(exist_ok=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
            "    'jax[cuda12]==0.11.2', 'optax==0.2.8', 'flax==0.12.10',\n"
            "    'numpy==2.5.4', 'tokenizers==0.23.3', 'pytest', 'hypothesis',\n"
            "    'matplotlib', 'datasets', 'optuna', 'tqdm', 'jaxtyping'], check=True)\n"
            "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'\n"
            "os.environ['PYTHONPATH'] = str(root / 'src')\n"
            "os.chdir(root / 'src')\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', str(root / 'src')], check=True)\n"
            "subprocess.run([sys.executable, '-c', 'import jax; print(jax.__version__, jax.devices())'], check=True)\n"
        ),
        code(
            "# Unit and tiny-model checks run on CPU, independent of the GPU diagnostic.\n"
            "test_env = dict(os.environ, JAX_PLATFORMS='cpu')\n"
            "checked = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_anchored_pc.py',\n"
            "    '-q', '-p', 'no:cacheprovider'], env=test_env, capture_output=True, text=True, timeout=300)\n"
            "print(checked.stdout, checked.stderr)\n"
            "checked.check_returncode()\n"
            "probe = subprocess.run([sys.executable, '-c', 'import jax; print(jax.__version__, jax.devices())'], capture_output=True, text=True, check=True)\n"
            "print(probe.stdout, probe.stderr)\n"
        ),
        code(
            "# Stage A only. Incremental evidence survives a failed scientific gate.\n"
            "import datetime, threading\n"
            "run_dir = root / ('results_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))\n"
            "command = [sys.executable, '-u', '-m', 'fabricpc.experiments.anchored_pc',\n"
            "    '--checkpoint', str(root / 'checkpoints/bert_mini_128.json'),\n"
            "    '--data-dir', str(root / 'data/domain_forgetting'),\n"
            "    '--output-dir', str(run_dir), '--budget-seconds', '7140']\n"
            "(root / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))\n"
            "process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "watchdog = threading.Timer(7200, process.kill)\n"
            "watchdog.daemon = True\n"
            "watchdog.start()\n"
            "try:\n"
            "    with (root / 'process.log').open('w') as log:\n"
            "        for line in process.stdout:\n"
            "            log.write(line)\n"
            "            log.flush()\n"
            "            print(line, end='', flush=True)\n"
            "    print('Process exit code:', process.wait())\n"
            "finally:\n"
            "    watchdog.cancel()\n"
            "summary = json.loads((run_dir / 'summary.json').read_text())\n"
            "print(json.dumps({k: summary.get(k) for k in ['status','decision','reason','elapsed_seconds','selected_solver']}, indent=2))\n"
        ),
        code(
            "# Export complete or partial evidence, source hashes and environment.\n"
            "import shutil\n"
            "shutil.copy(root / 'environment.txt', run_dir / 'environment.txt')\n"
            "shutil.copy(root / 'bundle_manifest.json', run_dir / 'bundle_manifest.json')\n"
            "shutil.copy(root / 'process.log', run_dir / 'process.log')\n"
            "(run_dir / 'colab_tests.log').write_text(checked.stdout + checked.stderr)\n"
            "hardware = subprocess.run(['nvidia-smi'], capture_output=True, text=True)\n"
            "(run_dir / 'hardware.txt').write_text(hardware.stdout + hardware.stderr)\n"
            "evidence = shutil.make_archive(str(run_dir), 'zip', run_dir)\n"
            "files.download(evidence)\n"
        ),
    ]
    notebook = output / "anchored_pc_stage_a.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "colab": {"name": notebook.name},
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "accelerator": "GPU",
                },
                "cells": cells,
            },
            indent=2,
        )
    )
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    return archive, notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for result in package(
        Path(__file__).resolve().parents[1], args.assets, args.output
    ):
        print(result)
