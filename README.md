# Graph Partitioning using GPU

## Preprocessing

This script is used to convert graph files in various formats into multiple useful output representations including CSR format and METIS-compatible graphs.

---

## 🧠 Purpose

Convert supported graph files (`.tsv`, `.txt`, `.mtx`, `.mmio`) into:
- CSR format: `row.txt`, `column.txt`, `weight.txt`
- METIS format: `temp.graph`
- Combined CSR file: `output_output.csr`

---

## 🚀 How to Run

```bash
python3 conversion.py <graph_filename>
```

### Example:

```bash
python3 conversion.py Graph.tsv
```

---

## 📥 Supported Input Formats

| Extension | Description                  | Notes                                |
|-----------|------------------------------|--------------------------------------|
| `.tsv`    | Tab-separated with 3 columns | Format: Source, Dest, Data           |
| `.txt`    | Tab-separated edge list      | Format: Source, Dest (skip 4 rows)   |
| `.mtx`    | Matrix Market format         | Reads via `scipy.io.mmread`         |
| `.mmio`   | Same as `.mtx`               | Alias extension                      |
| `.tsv_1`  | Tab-separated variant        | Format: Source, Dest (skip 1 row)    |

---

## 📤 Generated Outputs

| Output File        | Description                                        |
|--------------------|----------------------------------------------------|
| `row.txt`          | CSR row pointer list (1 entry per node + 1)        |
| `column.txt`       | CSR column index list (1 entry per edge)           |
| `weight.txt`       | List of edge weights (default: 1 for all edges)    |
| `output_output.csr`| Combined file of CSR structure and graph metadata  |
| `temp.graph`       | GPMetis-compatible graph (1-based node indices)    |

---

## 🧪 Output Preview (for a graph with 5 nodes and 6 edges)

```
> cat row.txt
0
2
4
4
5
6

> cat column.txt
1
2
0
3
4
2

> cat weight.txt
1
1
1
1
1
1
```

---

## 💡 Additional Info

- **Memory Usage Logging**: Current usage is printed at runtime.
- **Timing Info**: Prints the time taken for loading and converting the graph.
- **Self-loop Removal**: Automatically removes self-loops.
- **Bidirectional Conversion**: Converts the graph to undirected format.
- **Isolated Node Removal**: Removes nodes with no incoming or outgoing edges.

---

## 📦 Dependencies

Ensure the following Python libraries are installed:

```bash
pip install numpy pandas dgl torch tqdm scipy psutil
```

---

## 🛠️ Author Notes

- Designed for preprocessing graphs for partitioning, training GNNs, or CSR-based CUDA algorithms.
- Easily extendable to handle weighted inputs or directed graphs as needed.

---

# GPU Partitioning - CUDA Code Usage Guide

This guide provides instructions and helpful information on how to compile and run the `gpu_part.cu` file, along with details on expected inputs and generated outputs.

---

## ⚙️ Compilation

To compile the CUDA source file, use `nvcc`:

```bash
nvcc -O3 -arch=sm_60 gpu_part.cu -o gpu_part
```

Replace `sm_60` with the compute capability of your GPU (e.g., `sm_75` for NVIDIA Turing GPUs).

---

## ▶️ Running the Program

```bash
./gpu_part <input_row_file> <input_col_file> <num_partitions>
```

### Example:

```bash
./gpu_part row.txt column.txt 4
```

---

## 📥 Inputs

- `row.txt`: A text file containing the CSR row pointer array of the graph.
- `column.txt`: A text file containing the CSR column indices array.
- `num_partitions`: An integer specifying into how many parts the graph should be partitioned.

> 💡 These files are usually generated from a preprocessing script like `conversion.py`.

---

## 📤 Outputs

- `output.graph.part.<num_partitions>`: Contains the initial partition assignments for each vertex, as computed by METIS during the initial partitioning phase.
- `finalpartition.txt` : Contains the final, refined partition assignments for each vertex after GPU-accelerated deterministic refinement, optimizing edge cut and partition balance.

---

## 📝 Notes

- Make sure the input CSR files represent a valid graph with consistent sizes.
- Ensure that the number of partitions is a positive integer and smaller than or equal to the number of nodes.
- For large graphs, monitor GPU memory usage to avoid `cudaMalloc` failures.

---

## 🚀 Tips

- Use `nvprof` or `nsys` to profile performance.
- Enable compiler flags like `-lineinfo` for better debugging.
- Run on a machine with a CUDA-compatible GPU.

---

> ❗ **Disclaimer**: The actual functionality (e.g., partitioning algorithm used) depends on the implementation in `gpu_part.cu`.
