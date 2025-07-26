from torch import nn
from transformers import AutoTokenizer, EsmForProteinFolding

class EsmFold(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    def forward(self, seqs):
        inputs = self.tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        folded_positions = outputs.positions
        last_iteration = folded_positions[-1]
        backbone_coords = last_iteration[:, :, :4, :]
        backbone_coords = backbone_coords.reshape(backbone_coords.shape[0], -1, 3)
        return backbone_coords