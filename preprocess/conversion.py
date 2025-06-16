from numpy import array
import numpy as np
import pandas as pd
import sys
import dgl
import time
import torch as th
from scipy.io import mmread
import os
from tqdm import tqdm
import psutil

totalTime = 0
start = time.time()

file_name, file_extension = os.path.splitext(sys.argv[1])
print(f"File extension: {file_extension}")

# Output filenames
out_filename_row = "row.txt"
out_filename_col = "column.txt"
out_filename_weight = "weight.txt"
out_filename_metis = "temp.graph"
out_filename1 = "output_output.csr"  # Optional custom format

mem_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
print(f"Current memory usage: {mem_usage:.3f} GB")

# ----------- Load Graph ------------
if file_extension == '.mtx' or file_extension == '.mmio':
    print("Loading Matrix Market file...")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.graph((v, u))

elif file_extension == '.tsv':
    print("Loading TSV file...")
    columns = ['Source', 'Dest', 'Data']
    file = pd.read_csv(sys.argv[1], delimiter='\t', names=columns)
    G = dgl.graph((file['Source'].to_numpy(), file['Dest'].to_numpy()))

elif file_extension == '.txt':
    print("Loading TXT file...")
    columns = ['Source', 'Dest']
    file = pd.read_csv(sys.argv[1], delimiter='\t', names=columns, skiprows=4)
    G = dgl.graph((file['Source'].to_numpy(), file['Dest'].to_numpy()))

elif file_extension == '.tsv_1':
    print("Loading TSV_1 file...")
    columns = ['Source', 'Dest']
    file = pd.read_csv(sys.argv[1], delimiter='\t', names=columns, low_memory=False, skiprows=1)
    G = dgl.graph((file['Source'].to_numpy(), file['Dest'].to_numpy()))

else:
    print(f"Unsupported file type: {file_extension}")
    sys.exit("Supported: .tsv, .txt, .mtx, .mmio")

end = time.time()
print(f"Graph Loaded. Time: {round(end - start, 4)}s")

# ----------- Graph Preprocessing ------------
start = time.time()

G = dgl.remove_self_loop(G)
G = dgl.to_bidirected(G)
isolated_nodes = ((G.in_degrees() == 0) & (G.out_degrees() == 0)).nonzero().squeeze(1)
G.remove_nodes(isolated_nodes)

Nodes = G.num_nodes()
Edges = G.num_edges()
row_ptr = np.array(G.adj_sparse('csr')[0])
col_idx = np.array(G.adj_sparse('csr')[1])

print(f"Nodes: {Nodes}, Edges: {Edges}")
print(f"CSR size: row_ptr={len(row_ptr)}, col_idx={len(col_idx)}")

end = time.time()
print(f"Graph construction done. Time: {round(end - start, 4)}s")

# ----------- Save row_ptr, col_idx, weights ------------
with open(out_filename_row, 'w') as f:
    for r in row_ptr:
        f.write(f"{r}\n")

with open(out_filename_col, 'w') as f:
    for c in col_idx:
        f.write(f"{c}\n")

with open(out_filename_weight, 'w') as f:
    for _ in col_idx:
        f.write("1\n")

# ----------- Save custom CSR format (optional) ------------
with open(out_filename1, 'w') as f:
    f.write(f"{Nodes} {Edges} {len(row_ptr)} {len(col_idx)}\n")
    for r in tqdm(row_ptr, desc="Writing row_ptr"):
        f.write(f"{r} ")
    f.write("\n")
    for c in tqdm(col_idx, desc="Writing col_idx"):
        f.write(f"{c} ")

# ----------- Save GPMetis format (undirected edges) ------------
with open(out_filename_metis, 'w') as f:
    f.write(f"{Nodes} {Edges // 2} 1\n")  # Half the edges (bidirected counted once)
    for i in range(Nodes):
        neighbors = col_idx[row_ptr[i]:row_ptr[i + 1]]
        line = " ".join([f"{n + 1} 1" for n in neighbors])  # 1-based index
        f.write(line + "\n")
