import json

import torch
# import biotite
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from biotite.structure import lddt, tm_score, filter_canonical_amino_acids
from biotite.structure.filter import _filter_atom_names


from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder

if __name__ == '__main__':
    device = "cuda"
    model = FoldDecoder(device=device)
    token_data = {}

    with open("data/casp14_test/casp14_tokens.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            token_data.update(entry)

    token_example = token_data["T1082-D1"]["vqid"]
    vq_codes = torch.tensor(token_example, dtype=torch.long, device=device)
    protein=model.decode_single_prot(vq_codes,"test.pdb")
    X, _, _ = protein.to_XCS(all_atom=False)
    #print(X)
    #X_pred, all_preds = model.decode_to_structure(vq_codes)
    #print("\n\nx pred")
    #print(X_pred)
    #print("\n\nall_pred")
    #print(all_preds)

    # print("in project encoded tokens")
    #print(model.encode_pdb("tokenizer_benchmark/casps/casp14/T1082-D1.pdb"))
    # print("previous encoded tokens")
    # print(vq_codes)
    
    # get comaparision
    file = PDBFile.read("tokenizer_benchmark/casps/casp14/T1082-D1.pdb")
    array_stack = file.get_structure(model=1)
    array_stack = array_stack[_filter_atom_names(array_stack,["N", "CA", "C","O"])]
    
    #convert pred
    print(X)
    print("="*11)
    pred=X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
    print(pred)
    print("="*11)

    print(lddt(array_stack, pred,aggregation="residue"))

