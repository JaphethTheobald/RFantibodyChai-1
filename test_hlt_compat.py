from collections import OrderedDict

import torch

from rfantibody.rf2.modules.pose_util import CDR, Pose
from rfantibody.rf2.modules.preprocess import make_RF_idx


def make_pose():
    # H(2), L(1), T(3): standard internal H-L-T order.
    n = 6
    h = torch.tensor([True, True, False, False, False, False])
    l = torch.tensor([False, False, True, False, False, False])
    t = torch.tensor([False, False, False, True, True, True])
    cdrs = CDR(**{name: torch.zeros(n, dtype=torch.bool)
                  for name in ("H1", "H2", "H3", "L1", "L2", "L3")})
    return Pose(
        xyz=torch.zeros((n, 27, 3)),
        seq=torch.zeros(n, dtype=torch.long),
        atom_mask=torch.ones((n, 27), dtype=torch.bool),
        cdrs=cdrs,
        idx=torch.tensor([31, 32, 101, 700, 701, 702]),
        chain_dict=OrderedDict(H=h, L=l, T=t),
        hotspots=torch.zeros(n, dtype=torch.bool),
    )


pose = make_pose()
same = pose.same_chain

# H/L is one antibody group; T is the other.  All target-antibody pairs, and
# only those pairs, must be selected by ~same_chain.
assert same[:3, :3].all()
assert same[3:, 3:].all()
assert not same[:3, 3:].any()
assert not same[3:, :3].any()

# Each chain begins at zero before the next chain receives 200 unused indices.
assert torch.equal(
    make_RF_idx(pose),
    torch.tensor([0, 1, 202, 403, 404, 405]),
)

print("HLT compatibility checks passed")
