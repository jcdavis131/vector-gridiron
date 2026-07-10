"""Vector Gridiron MTNN — compatibility entrypoint.

v1 was a flat shared-trunk multi-task net. v2 lives in `train_mtnn.py`
(multi-tower gated fusion). This module re-exports that path so older
scripts keep working:

  python pipeline/train_models.py   ->  train_mtnn.main()
  python pipeline/train_mtnn.py     ->  preferred explicit entry
"""

from __future__ import annotations

import train_mtnn


def main():
    return train_mtnn.main()


if __name__ == "__main__":
    raise SystemExit(main())
