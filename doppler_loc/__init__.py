"""Locate an audio recording from the Doppler curves of passing aircraft."""

import os

# The work is shared out over worker processes, so the maths library needn't start threads of its own.
# Set before NumPy and SciPy load: OpenBLAS reserves buffers for every thread it might start, in each
# copy of it (they ship one each), which was 1 GB per process on 16 threads, or 9 GB across a search.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
