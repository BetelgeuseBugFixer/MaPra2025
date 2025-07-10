def _masked_accuracy(logits, tgt, mask):
    pred = logits.argmax(dim=-1)
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0

def calc_lddt_scores(protein_predictions, ref_protein):
    lddt_scores = []
    for protein_prediction, protein_ref in zip(protein_predictions, ref_protein):
        X, _, _ = protein_prediction.to_XCS(all_atom=False)
        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        lddt_scores.append(lddt(protein_ref, X))
    return lddt_scores


def calc_token_loss(criterion, tokens_predictions, tokens_reference):
    return criterion(tokens_predictions.transpose(1, 2), tokens_reference)