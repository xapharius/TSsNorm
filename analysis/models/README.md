# Importance matrices

Drop `[tasks, filters]` importance matrices here — the demo notebook loads
`nyuv2_tssbn.npy` from this directory.

Save one from any trained TSsBN model:

```python
import numpy as np
from tssnorm import get_importance

np.save("analysis/models/my_model.npy", get_importance(model).cpu().numpy())
```

Matrices are not tracked by git.
