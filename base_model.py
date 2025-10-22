import torch
import torch.nn as nn
import torch.nn.functional as F

from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        vggt4track_model.eval()
        vggt4track_model = vggt4track_model.to(device)

    def forward(self, x):
        # Define the forward pass
        # x = self.conv1(x)
        return x
