import json
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file = self.log_dir / f"train_{ts}.log"
        self._fp = self.file.open("w", encoding="utf-8")

    def log_config(self, cfg: dict):
        self._fp.write("# CONFIG\n")
        self._fp.write(json.dumps(cfg, indent=2) + "\n")
        self._fp.flush()

    def log_round(self, r, eval_res, comm_pct):
        line = {
            "round": int(r),
            "avg_reward": float(eval_res["avg_reward"]),
            "accuracy_pct": float(eval_res["accuracy_pct"]),
            "comm_overhead_pct": float(comm_pct)
        }
        self._fp.write(json.dumps(line) + "\n")
        self._fp.flush()
        print(f'[Round {r}] reward={line["avg_reward"]:.1f} acc={line["accuracy_pct"]:.1f}% comm={line["comm_overhead_pct"]:.1f}%')

    def log_final(self, eval_res, comm_pct):
        self._fp.write("# FINAL\n")
        self._fp.write(json.dumps({"final_eval": eval_res, "avg_comm_overhead_pct": float(comm_pct)}, indent=2) + "\n")
        self._fp.flush()
        print(f'[FINAL] reward={eval_res["avg_reward"]:.1f} acc={eval_res["accuracy_pct"]:.1f}% avg_comm={comm_pct:.1f}%')

    def __del__(self):
        try:
            self._fp.close()
        except Exception:
            pass
