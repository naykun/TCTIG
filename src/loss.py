import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max
        # print("scores:---max:", scores.max(), "---min:", scores.min())
        # print("targets:---max:", targets.max(), "---min:", targets.min())

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, "caption_len"):
            # caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            caption_lengths, _ = sample_list.caption_len
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True, enforce_sorted=False
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # import ipdb; ipdb.set_trace()
        loss = F.cross_entropy(scores, targets)

        return loss
    
class ImageCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max
        # print("scores:---max:", scores.max(), "---min:", scores.min())
        # print("targets:---max:", targets.max(), "---min:", targets.min())
        
        decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True, enforce_sorted=False
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # import ipdb; ipdb.set_trace()
        loss = F.cross_entropy(scores, targets)

        return loss