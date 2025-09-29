from typing import List, Dict
import torch

class FedProxAggregator:
    """
    Aggregation identical to FedAvg; proximal term is added in local loss
    (handled inside Agent). This class exists for symmetry with CLI.
    """
    def aggregate(self, client_updates: List[Dict]) -> torch.Tensor:
        num_params = client_updates[0]["num_params"]
        acc = torch.zeros(num_params, dtype=torch.float32)
        cnt = 0
        for u in client_updates:
            if u["type"] == "dense":
                acc += u["delta"]; cnt += 1
            elif u["type"] == "sparse":
                tmp = torch.zeros(num_params, dtype=torch.float32)
                tmp[u["indices"]] = u["values"]
                acc += tmp; cnt += 1
        if cnt > 0:
            acc /= float(cnt)
        return acc
