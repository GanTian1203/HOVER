# Environments

| Use | Recorded environment |
| --- | --- |
| Formal training | Python 3.8.20, PyTorch 2.0.0+cu118, NumPy 1.23.5 |
| Clean CPU verification | Windows, Python 3.8.20, PyTorch 2.0.0+cpu, torchvision 0.15.1+cpu, NumPy 1.23.5 |
| Limited GPU verification | Original cu118 framework in a separate environment |

Install [requirements-cpu.txt](../requirements-cpu.txt) for CPU runs or
[requirements.txt](../requirements.txt) for the recorded CUDA stack. The
[CPU lock file](../requirements-cpu-lock.txt) and
[environment record](../manifests/phase3_clean_environment.json) list the
versions used in the clean CPU environment. That environment passed `pip check`,
the CPU regression suite, and an exact comparison of 98 synthetic arrays
against the earlier CPU framework build.

Windows PowerShell uses `.venv\Scripts\Activate.ps1`. Linux shells use
`source .venv/bin/activate`. CLI paths can be relative to the repository.
Linux, newer Python versions, and a fresh CUDA installation are outside the
verified environments. See [reproduction checks](VERIFICATION_STATUS.md) for
the GPU test scope.
