from typing import List, Dict
import torch

class AEFCAggregator:
    """
    AEFC server-side aggregator:
    - expects sparse client updates + credibility scores
    - reconstructs dense global delta via credibility-weighted sum
    """
    def __init__(self, kappa: float = 0.2):
        assert 0.0 < kappa <= 1.0
        self.kappa = kappa

    def aggregate(self, client_updates: List[Dict]) -> torch.Tensor:
        """Return dense delta vector for global actor parameters."""
        assert len(client_updates) > 0
        num_params = client_updates[0]["num_params"]
        out = torch.zeros(num_params, dtype=torch.float32)

        # collect positive cred; avoid all-zero
        creds = []
        for u in client_updates:
            if u["type"] == "sparse":
                creds.append(max(float(u.get("cred", 0.0)), 0.0))
            else:
                creds.append(0.0)
        s = sum(creds)
        weights = [ (c / s) if s > 0.0 else 0.0 for c in creds ]

        # if all weights zero (e.g., all dense or all non-positive cred), fallback to equal average of dense parts
        fallback = (sum(weights) == 0.0)

        if fallback:
            # treat everything as dense average
            cnt = 0
            for u in client_updates:
                if u["type"] == "dense":
                    out += u["delta"]
                    cnt += 1
                elif u["type"] == "sparse":
                    # sparse -> make dense then average
                    tmp = torch.zeros(num_params, dtype=torch.float32)
                    tmp[u["indices"]] = u["values"]
                    out += tmp
                    cnt += 1
            if cnt > 0:
                out /= float(cnt)
            return out

        # weighted sum (sparse -> dense then weighted add)
        for w, u in zip(weights, client_updates):
            if w == 0.0:
                continue
            if u["type"] == "sparse":
                out[u["indices"]] += w * u["values"]
            else:
                out += w * u["delta"]
        return out
