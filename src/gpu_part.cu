// Filename: 7thmay.cu
// Description: This CUDA program implements a GPU-accelerated graph partitioning algorithm.
// It uses a multi-phase approach: coarsening the graph, initial partitioning with METIS,
// and deterministic refinement to minimize edge cut while maintaining balance constraints.

// Includes necessary headers for I/O, vectors, CUDA runtime, and Thrust for GPU operations.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <set>
#include <cmath>
#include <iomanip>

// Defines constants for CUDA thread block size and algorithm parameters.
#define BLOCK_SIZE 256  // Number of threads per block for CUDA kernels.
#define MAX_REFINEMENT_PASSES 40  // Maximum iterations for refinement phase.
#define MAX_BALANCE_PASSES 40  // Maximum iterations for balance phase.
#define MAX_IMBALANCE 0.03f // Maximum allowed partition imbalance (3%).

// Struct to represent a vertex move between partitions during refinement.
struct Move {
    int vertexId;      // ID of the vertex to move.
    int source;        // Source partition.
    int destination;   // Target partition.
    float gain;        // Gain from moving the vertex.
};

// Macro to check CUDA errors and exit on failure with an error message.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==============================
// Coarsening Kernels
// ==============================
// These kernels handle the graph coarsening phase by matching vertices and reducing graph size.

// Kernel to match vertices based on maximum edge weights for coarsening.
__global__ void matchVertices(int* xadj, int* adjncy, int* adjwgt, int* match, int* tempweight, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices || match[tid] != -1) return;
    int start = xadj[tid];
    int end = xadj[tid + 1];
    int bestMatch = -1;
    int maxWeight = -1;
    for (int i = start; i < end; i++) {
        int neighbor = adjncy[i];
        int weight = adjwgt[i];
        if (neighbor < numVertices && weight > maxWeight && match[neighbor] == -1 && neighbor != tid) {
            maxWeight = weight;
            bestMatch = neighbor;
        }
    }
    if (bestMatch != -1) {
        int expected = -1;
        if (atomicCAS(&match[bestMatch], expected, tid) == expected) {
            match[tid] = bestMatch;
            tempweight[tid] = maxWeight;
            tempweight[bestMatch] = maxWeight;
        }
    }
}

// Kernel to rematch unmatched vertices or refine initial matches.
__global__ void rematchVertices(int* xadj, int* adjncy, int* adjwgt, int* match, int* tempweight, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices || (match[tid] != -1 && tid == match[match[tid]])) return;
    match[tid] = -1;
    tempweight[tid] = -1;
    int start = xadj[tid];
    int end = xadj[tid + 1];
    int bestMatch = -1;
    int maxWeight = -1;
    for (int i = start; i < end; i++) {
        int neighbor = adjncy[i];
        int weight = adjwgt[i];
        if (neighbor < numVertices && weight > maxWeight && match[neighbor] == -1 && neighbor != tid) {
            maxWeight = weight;
            bestMatch = neighbor;
        }
    }
    if (bestMatch != -1) {
        int expected = -1;
        if (atomicCAS(&match[bestMatch], expected, tid) == expected) {
            match[tid] = bestMatch;
            tempweight[tid] = maxWeight;
            tempweight[bestMatch] = maxWeight;
        }
    }
}

// Kernel to map matched vertices to new vertex IDs in the coarsened graph.
__global__ void mapVertices(int* match, int* cmap, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices) return;
    if (match[tid] != -1 && tid < match[tid]) {
        cmap[tid] = tid;
        cmap[match[tid]] = tid;
    } else if (match[tid] == -1) {
        cmap[tid] = tid;
    }
}

// Kernel to relabel coarse vertices and create a traceback array for uncoarsening.
__global__ void relabel_coarse_vertices_kernel(int* CMap, int* New_CMap, int* label_map, int* traceBack, int num_elements, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements || CMap[idx] != idx) return;
    int new_id = atomicAdd(count, 1);
    label_map[idx] = new_id;
    New_CMap[idx] = new_id;
    traceBack[2 * new_id] = idx;
    int match = -1;
    for (int i = idx + 1; i < num_elements; i++) {
        if (CMap[i] == idx) {
            match = i;
            break;
        }
    }
    if (match != -1) {
        traceBack[2 * new_id + 1] = match;
        New_CMap[match] = new_id;
    } else {
        traceBack[2 * new_id + 1] = -1;
    }
}

// Kernel to count unique edges in the coarsened graph using a hash-based approach.
__global__ void countUniqueEdges(int* rowPtr, int* colInd, int* weights, int* newIDs, int* matches, int* cmap, int* uniqueCounts, const int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    int orig_idx = idx;
    idx = cmap[idx];
    unsigned long long foundHash = 0;
    int uniqueCount = 0;
    auto checkUnique = [&](int neighbor) {
        unsigned long long mask = 1ULL << (neighbor % 64);
        if (!(foundHash & mask)) {
            foundHash |= mask;
            uniqueCount++;
        }
    };
    for (int i = rowPtr[orig_idx]; i < rowPtr[orig_idx + 1]; i++) {
        int neighbor = newIDs[colInd[i]];
        if (neighbor != newIDs[orig_idx] && neighbor != -1) {
            checkUnique(neighbor);
        }
    }
    int matchedVertex = matches[orig_idx];
    if (matchedVertex != -1 && matchedVertex < numVertices) {
        for (int i = rowPtr[matchedVertex]; i < rowPtr[matchedVertex + 1]; i++) {
            int matchNeighbor = newIDs[colInd[i]];
            if (matchNeighbor != newIDs[orig_idx] && matchNeighbor != -1) {
                checkUnique(matchNeighbor);
            }
        }
    }
    uniqueCounts[orig_idx] = uniqueCount;
}

// Kernel to compute prefix sum for row pointers in CSR format of the coarsened graph.
__global__ void prefix_sum(int* row_ptr, int* updaterow, int nvtxs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        updaterow[0] = 0;
        for (int i = 1; i <= nvtxs; i++) {
            updaterow[i] = updaterow[i - 1] + row_ptr[i - 1];
        }
    }
}

// Kernel to fill edge arrays (column indices and weights) for the coarsened graph.
__global__ void fillEdgeArrays(int* rowPtr, int* colInd, int* weights, int* newIDs, int* matches, int* cmap, int* outputRowPtr, int* outputColInd, int* outputWeights, const int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices) return;
    int orig_tid = tid;
    tid = cmap[tid];
    int match = matches[orig_tid];
    int currentOutputPos = outputRowPtr[orig_tid];
    unsigned long long foundHash = 0;
    auto addUniqueEdge = [&](int neighbor, int weight) {
        unsigned long long mask = 1ULL << (neighbor % 64);
        if (!(foundHash & mask)) {
            if (currentOutputPos < outputRowPtr[orig_tid + 1]) {
                outputColInd[currentOutputPos] = neighbor;
                outputWeights[currentOutputPos] = weight;
                currentOutputPos++;
                foundHash |= mask;
            }
        } else {
            for (int j = outputRowPtr[orig_tid]; j < currentOutputPos; j++) {
                if (outputColInd[j] == neighbor) {
                    outputWeights[j] += weight;
                    break;
                }
            }
        }
    };
    for (int i = rowPtr[orig_tid]; i < rowPtr[orig_tid + 1]; i++) {
        int neighbor = newIDs[colInd[i]];
        int weight = weights[i];
        if (neighbor != newIDs[orig_tid] && neighbor != -1) {
            addUniqueEdge(neighbor, weight);
        }
    }
    if (match != -1 && match < numVertices) {
        for (int i = rowPtr[match]; i < rowPtr[match + 1]; i++) {
            int neighbor = newIDs[colInd[i]];
            int weight = weights[i];
            if (neighbor != newIDs[orig_tid] && neighbor != -1) {
                addUniqueEdge(neighbor, weight);
            }
        }
    }
}

// Kernel to update row pointers for the coarsened graph's CSR format.
__global__ void updateRowPtr(int* rowPtr, int* newRowPtr, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        newRowPtr[0] = 0;
        for (int i = 1; i <= numVertices; i++) {
            newRowPtr[i] = rowPtr[i];
        }
    }
}

// Kernel to reset the match array to -1 for the next coarsening iteration.
__global__ void resetMatch(int* match, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        match[tid] = -1;
    }
}

// ==============================
// Refinement Kernels
// ==============================
// These kernels refine the partition by moving vertices to reduce edge cut and balance partitions.

// Kernel to compute gains and best target partitions for each vertex move.
__global__ void computeGainsAndDestinations(int numVertices, int numPartitions, const int* xadj, const int* adjncy, const int* edgeWeight, const int* partition, float* gain, int* target, int nedges, const int* partCount, int targetSize, bool balancePhase, float* counts) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVertices) return;
    int curPart = partition[v];
    if (curPart >= numPartitions || curPart < 0) return;

    extern __shared__ float shared_counts[];
    float* counts_ptr;
    if (numPartitions <= 32) {
        float local_counts[32] = {0};
        counts_ptr = local_counts;
    } else {
        counts_ptr = &counts[v * numPartitions];
        for (int q = 0; q < numPartitions; q++) {
            counts_ptr[q] = 0.0f;
        }
    }

    int start = xadj[v];
    int end = (v + 1 < numVertices) ? xadj[v + 1] : nedges;
    for (int i = start; i < end; i++) {
        int nbr = adjncy[i];
        if (nbr < numVertices) {
            int nbrPart = partition[nbr];
            if (nbrPart < numPartitions && nbrPart >= 0) counts_ptr[nbrPart] += edgeWeight[i];
        }
    }

    float bestGain = -1e9f;
    int bestTarget = curPart;
    float srcSize = (float)partCount[curPart];
    for (int q = 0; q < numPartitions; q++) {
        if (q != curPart) {
            float edgeGain = counts_ptr[q] - counts_ptr[curPart];
            float destSize = (float)partCount[q];
            float balanceGain = balancePhase ? 500.0f * (targetSize - destSize) / targetSize : 200.0f * (targetSize - destSize) / targetSize;
            if (srcSize > targetSize * 1.05f) balanceGain += balancePhase ? 250.0f : 100.0f;
            if (destSize < targetSize * 0.95f) balanceGain += balancePhase ? 125.0f : 50.0f;
            float g = edgeGain + balanceGain;
            if (g > bestGain || (g == bestGain && q < bestTarget)) {
                bestGain = g;
                bestTarget = q;
            }
        }
    }
    gain[v] = bestGain;
    target[v] = bestTarget;
}

// Kernel to mark vertices as move candidates based on gain thresholds.
__global__ void markCandidates(int numVertices, const float* gain, bool* candidate, bool balancePhase) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVertices) return;
    candidate[v] = balancePhase ? (gain[v] > -50.0f) : (gain[v] > -10.0f);
}

// Kernel to select move candidates using a token-passing conflict resolution mechanism.
__global__ void tokenPassing(int numVertices, const int* xadj, const int* adjncy, const float* gain, const bool* candidate, bool* finalCandidate, int nedges) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVertices || !candidate[v]) {
        if (v < numVertices) finalCandidate[v] = false;
        return;
    }
    bool win = true;
    int start = xadj[v];
    int end = (v + 1 < numVertices) ? xadj[v + 1] : nedges;
    for (int i = start; i < end; i++) {
        int u = adjncy[i];
        if (u < numVertices && candidate[u] && ((gain[u] > gain[v] + 10.0f) || (gain[u] == gain[v] && u < v))) {
            win = false;
            break;
        }
    }
    finalCandidate[v] = win;
}

// Kernel to buffer selected candidate moves into a move buffer.
__global__ void bufferCandidateMoves(int numVertices, const int* partition, const int* target, const float* gain, const bool* finalCandidate, Move* moveBuffer, int* globalMoveCount, int* outCounter, int* inCounter, int maxMoves, int numPartitions) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVertices || !finalCandidate[v]) return;
    int src = partition[v];
    int dest = target[v];
    if (src < numPartitions && dest < numPartitions && src != dest && src >= 0 && dest >= 0) {
        int pos = atomicAdd(globalMoveCount, 1);
        if (pos < maxMoves) {
            moveBuffer[pos] = {v, src, dest, gain[v]};
            atomicAdd(&outCounter[src], 1);
            atomicAdd(&inCounter[dest], 1);
        }
    }
}

// Device function to compute the minimum of two integers.
__device__ inline int cudaMin(int a, int b) {
    return (a < b) ? a : b;
}

// Kernel to compute available capacity in each partition based on balance constraints.
__global__ void computeAvailableCapacity(int numPartitions, const int* partCount, const int* outCounter, const int* inCounter, int* avail, int targetSize, float maxImbalance, bool balancePhase, int nvtxs) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= numPartitions) return;
    int current = partCount[q];
    int slack = max(100, nvtxs / (numPartitions * 10));
    int maxSize = targetSize + (int)(targetSize * maxImbalance) + slack;
    int minSize = targetSize - (int)(targetSize * maxImbalance) - slack;
    avail[q] = targetSize - current + outCounter[q] + slack;
    if (avail[q] < 0) avail[q] = 0;
    if (avail[q] > maxSize - current + slack) avail[q] = maxSize - current + slack;
    if (current <= minSize && !balancePhase) avail[q] = cudaMin(avail[q], slack / 2);
}

// Kernel to select moves based on available partition capacity.
__global__ void selectMoves(int totalMoves, Move* moveBuffer, const int* avail, bool* selectedFlag, int* destCounts, int numPartitions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalMoves) return;
    Move m = moveBuffer[i];
    int dest = m.destination;
    int pos = atomicAdd(&destCounts[dest], 1);
    selectedFlag[i] = (pos < avail[dest]);
}

// Kernel to commit selected moves by updating partition assignments and counts.
__global__ void commitMoves(int totalMoves, Move* moveBuffer, const bool* selectedFlag, int* partition, int* partCount, int numPartitions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalMoves || !selectedFlag[i]) return;
    Move m = moveBuffer[i];
    if (m.source < numPartitions && m.destination < numPartitions && m.source >= 0 && m.destination >= 0) {
        int vertex = m.vertexId;
        int oldPart = atomicExch(&partition[vertex], m.destination);
        if (oldPart == m.source) {
            atomicAdd(&partCount[m.destination], 1);
            atomicAdd(&partCount[m.source], -1);
        } else {
            atomicExch(&partition[vertex], oldPart);
        }
    }
}

// Kernel to compute the edge cut by summing weights of edges crossing partitions.
__global__ void compute_edgecut(int* d_xadj, int* d_adjncy, int* d_adjwgt, int* d_partition, int* d_edgecut, int nvtxs, int nedges, int numPartitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int block_edgecut;
    if (threadIdx.x == 0) block_edgecut = 0;
    __syncthreads();
    if (idx < nvtxs) {
        int local_edgecut = 0;
        int start = d_xadj[idx];
        int end = (idx + 1 < nvtxs) ? d_xadj[idx + 1] : nedges;
        int part_idx = d_partition[idx];
        for (int k = start; k < end; k++) {
            int neighbor = d_adjncy[k];
            if (neighbor < nvtxs) {
                int part_nbr = d_partition[neighbor];
                if (part_idx != part_nbr && part_idx >= 0 && part_idx < numPartitions && part_nbr >= 0 && part_nbr < numPartitions) {
                    local_edgecut += d_adjwgt[k];
                }
            }
        }
        atomicAdd(&block_edgecut, local_edgecut);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(d_edgecut, block_edgecut);
}

// Function to get current time in seconds for timing measurements.
double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// Function to compute edge cut, including kernel launch and timing.
int computeEdgeCut(int nvtxs, int nedges, int* d_xadj, int* d_adjncy, int* d_adjwgt, thrust::device_vector<int>& d_partition, int numPartitions, float& edgeCutKernelTime, float& edgeCutOverheadTime, cudaEvent_t kernelStart, cudaEvent_t kernelStop) {
    double start = getTime();
    double eventOverheadStart = getTime();
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));
    edgeCutOverheadTime += getTime() - eventOverheadStart;

    int* d_edgecut;
    CUDA_CHECK(cudaMalloc(&d_edgecut, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edgecut, 0, sizeof(int)));

    int gridSize = (nvtxs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaEventRecord(kernelStart));
    compute_edgecut<<<gridSize, BLOCK_SIZE>>>(d_xadj, d_adjncy, d_adjwgt, thrust::raw_pointer_cast(d_partition.data()), d_edgecut, nvtxs, nedges, numPartitions);
    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaEventSynchronize(kernelStop));

    float kernelTimeMs;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
    edgeCutKernelTime += kernelTimeMs / 1000.0f;

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_edgecut;
    CUDA_CHECK(cudaMemcpy(&h_edgecut, d_edgecut, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_edgecut));

    eventOverheadStart = getTime();
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));
    edgeCutOverheadTime += getTime() - eventOverheadStart;

    edgeCutOverheadTime += getTime() - start - (kernelTimeMs / 1000.0f);
    return h_edgecut / 2;
}

// Kernel for one step of bitonic sort to sort moves by gain.
__global__ void bitonicSortStep(Move* data, int n, int k, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;
    if (ixj > idx || idx >= n || ixj >= n) return;
    bool up = ((idx & k) == 0);
    Move a = data[idx];
    Move b = data[ixj];
    bool swap = (a.gain < b.gain) ^ up;
    if (swap) {
        data[idx] = b;
        data[ixj] = a;
    }
}

// Function to sort moves in the move buffer using bitonic sort.
void customSortMoves(thrust::device_vector<Move>& d_moveBuffer, int h_totalMoves) {
    Move* d_data = thrust::raw_pointer_cast(d_moveBuffer.data());
    int n = h_totalMoves;
    int max_size = d_moveBuffer.size();
    int padded_n = 1;
    while (padded_n < n) padded_n <<= 1;
    if (padded_n > max_size) padded_n = max_size;
    if (padded_n > n) {
        thrust::fill(d_moveBuffer.begin() + n, d_moveBuffer.begin() + padded_n, Move{-1, -1, -1, -1e9f});
        n = padded_n;
    }
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<gridSize, BLOCK_SIZE>>>(d_data, n, k, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// Function to perform deterministic refinement with non-balance and balance phases.
void deterministicRefinement(int nvtxs, int nedges, int* d_xadj, int* d_adjncy, int* d_adjwgt, thrust::device_vector<int>& d_partition, thrust::device_vector<int>& d_partCount, int numPartitions, int& edgeCut, float& edgeCutKernelTime, float& refinementKernelTime, float& refinementOverheadTime) {
    double start = getTime();
    double eventOverheadStart = getTime();
    cudaEvent_t kernelStart, kernelStop;
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));
    refinementOverheadTime += getTime() - eventOverheadStart;
    float kernelTimeMs = 0.0f;

    int gridSize = (nvtxs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int maxMoves = nvtxs;

    thrust::device_vector<float> d_gain(nvtxs);
    thrust::device_vector<int> d_target(nvtxs);
    thrust::device_vector<bool> d_candidate(nvtxs);
    thrust::device_vector<bool> d_finalCandidate(nvtxs);
    thrust::device_vector<Move> d_moveBuffer(maxMoves);
    thrust::device_vector<bool> d_selectedFlag(maxMoves);
    thrust::device_vector<int> d_outCounter(numPartitions, 0);
    thrust::device_vector<int> d_inCounter(numPartitions, 0);
    thrust::device_vector<int> d_avail(numPartitions);
    thrust::device_vector<int> d_destCounts(numPartitions);
    thrust::device_vector<float> d_counts(nvtxs * numPartitions, 0.0f);

    int* d_globalMoveCount;
    CUDA_CHECK(cudaMalloc(&d_globalMoveCount, sizeof(int)));

    int bestEdgeCut = edgeCut;
    float bestBalanceScore = 1e9f;
    thrust::device_vector<int> d_bestPartition(nvtxs);
    thrust::copy(d_partition.begin(), d_partition.end(), d_bestPartition.begin());

    int targetSize = nvtxs / numPartitions;
    float maxImbalance = 0.05f;

    // Non-balance phase: Prioritize reducing edge cut.
    for (int pass = 0; pass < 30; pass++) {
        CUDA_CHECK(cudaMemset(d_globalMoveCount, 0, sizeof(int)));
        thrust::fill(d_outCounter.begin(), d_outCounter.end(), 0);
        thrust::fill(d_inCounter.begin(), d_inCounter.end(), 0);

        size_t sharedMemSize = (numPartitions <= 32) ? 0 : numPartitions * sizeof(float);
        CUDA_CHECK(cudaEventRecord(kernelStart));
        computeGainsAndDestinations<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(nvtxs, numPartitions, d_xadj, d_adjncy, d_adjwgt, thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_target.data()), nedges, thrust::raw_pointer_cast(d_partCount.data()), targetSize, false, thrust::raw_pointer_cast(d_counts.data()));
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        markCandidates<<<gridSize, BLOCK_SIZE>>>(nvtxs, thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_candidate.data()), false);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        tokenPassing<<<gridSize, BLOCK_SIZE>>>(nvtxs, d_xadj, d_adjncy, thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_candidate.data()), thrust::raw_pointer_cast(d_finalCandidate.data()), nedges);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        bufferCandidateMoves<<<gridSize, BLOCK_SIZE>>>(nvtxs, thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_target.data()), thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_finalCandidate.data()), thrust::raw_pointer_cast(d_moveBuffer.data()), d_globalMoveCount, thrust::raw_pointer_cast(d_outCounter.data()), thrust::raw_pointer_cast(d_inCounter.data()), maxMoves, numPartitions);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_totalMoves;
        CUDA_CHECK(cudaMemcpy(&h_totalMoves, d_globalMoveCount, sizeof(int), cudaMemcpyDeviceToHost));
        h_totalMoves = std::min(h_totalMoves, maxMoves);

        if (h_totalMoves > 0) {
            CUDA_CHECK(cudaEventRecord(kernelStart));
            computeAvailableCapacity<<<(numPartitions + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numPartitions, thrust::raw_pointer_cast(d_partCount.data()), thrust::raw_pointer_cast(d_outCounter.data()), thrust::raw_pointer_cast(d_inCounter.data()), thrust::raw_pointer_cast(d_avail.data()), targetSize, maxImbalance, false, nvtxs);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            customSortMoves(d_moveBuffer, h_totalMoves);

            thrust::fill(d_destCounts.begin(), d_destCounts.end(), 0);
            CUDA_CHECK(cudaEventRecord(kernelStart));
            selectMoves<<<(h_totalMoves + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(h_totalMoves, thrust::raw_pointer_cast(d_moveBuffer.data()), thrust::raw_pointer_cast(d_avail.data()), thrust::raw_pointer_cast(d_selectedFlag.data()), thrust::raw_pointer_cast(d_destCounts.data()), numPartitions);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(kernelStart));
            commitMoves<<<(h_totalMoves + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(h_totalMoves, thrust::raw_pointer_cast(d_moveBuffer.data()), thrust::raw_pointer_cast(d_selectedFlag.data()), thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_partCount.data()), numPartitions);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            float currentEdgeCutKernelTime = 0.0f;
            float currentEdgeCutOverheadTime = 0.0f;
            int currentEdgeCut = computeEdgeCut(nvtxs, nedges, d_xadj, d_adjncy, d_adjwgt, d_partition, numPartitions, currentEdgeCutKernelTime, currentEdgeCutOverheadTime, kernelStart, kernelStop);
            edgeCutKernelTime += currentEdgeCutKernelTime;

            thrust::host_vector<int> h_partCount = d_partCount;
            float balanceScore = 0.0f;
            for (int q = 0; q < numPartitions; q++) {
                float diff = (float)h_partCount[q] - targetSize;
                balanceScore += diff * diff;
            }
            balanceScore = sqrt(balanceScore / numPartitions);
            if (currentEdgeCut < bestEdgeCut || (currentEdgeCut <= bestEdgeCut * 1.01f && balanceScore < bestBalanceScore)) {
                bestEdgeCut = currentEdgeCut;
                bestBalanceScore = balanceScore;
                thrust::copy(d_partition.begin(), d_partition.end(), d_bestPartition.begin());
            }

            edgeCut = currentEdgeCut;
            if (h_totalMoves < nvtxs / 200 && thrust::count(d_selectedFlag.begin(), d_selectedFlag.begin() + h_totalMoves, true) < nvtxs / 400) {
                break;
            }
        } else {
            break;
        }
    }

    thrust::copy(d_bestPartition.begin(), d_bestPartition.end(), d_partition.begin());
    edgeCut = bestEdgeCut;

    // Balance phase: Prioritize balancing partitions within the imbalance constraint.
    bool balanced = false;
    for (int pass = 0; pass < 40 && !balanced; pass++) {
        CUDA_CHECK(cudaMemset(d_globalMoveCount, 0, sizeof(int)));
        thrust::fill(d_outCounter.begin(), d_outCounter.end(), 0);
        thrust::fill(d_inCounter.begin(), d_inCounter.end(), 0);

        size_t sharedMemSize = (numPartitions <= 32) ? 0 : numPartitions * sizeof(float);
        CUDA_CHECK(cudaEventRecord(kernelStart));
        computeGainsAndDestinations<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(nvtxs, numPartitions, d_xadj, d_adjncy, d_adjwgt, thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_target.data()), nedges, thrust::raw_pointer_cast(d_partCount.data()), targetSize, true, thrust::raw_pointer_cast(d_counts.data()));
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        markCandidates<<<gridSize, BLOCK_SIZE>>>(nvtxs, thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_candidate.data()), true);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        tokenPassing<<<gridSize, BLOCK_SIZE>>>(nvtxs, d_xadj, d_adjncy, thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_candidate.data()), thrust::raw_pointer_cast(d_finalCandidate.data()), nedges);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernelStart));
        bufferCandidateMoves<<<gridSize, BLOCK_SIZE>>>(nvtxs, thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_target.data()), thrust::raw_pointer_cast(d_gain.data()), thrust::raw_pointer_cast(d_finalCandidate.data()), thrust::raw_pointer_cast(d_moveBuffer.data()), d_globalMoveCount, thrust::raw_pointer_cast(d_outCounter.data()), thrust::raw_pointer_cast(d_inCounter.data()), maxMoves, numPartitions);
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        refinementKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_totalMoves;
        CUDA_CHECK(cudaMemcpy(&h_totalMoves, d_globalMoveCount, sizeof(int), cudaMemcpyDeviceToHost));
        h_totalMoves = std::min(h_totalMoves, maxMoves);

        if (h_totalMoves > 0) {
            CUDA_CHECK(cudaEventRecord(kernelStart));
            computeAvailableCapacity<<<(numPartitions + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numPartitions, thrust::raw_pointer_cast(d_partCount.data()), thrust::raw_pointer_cast(d_outCounter.data()), thrust::raw_pointer_cast(d_inCounter.data()), thrust::raw_pointer_cast(d_avail.data()), targetSize, maxImbalance, true, nvtxs);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            customSortMoves(d_moveBuffer, h_totalMoves);

            thrust::fill(d_destCounts.begin(), d_destCounts.end(), 0);
            CUDA_CHECK(cudaEventRecord(kernelStart));
            selectMoves<<<(h_totalMoves + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(h_totalMoves, thrust::raw_pointer_cast(d_moveBuffer.data()), thrust::raw_pointer_cast(d_avail.data()), thrust::raw_pointer_cast(d_selectedFlag.data()), thrust::raw_pointer_cast(d_destCounts.data()), numPartitions);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(kernelStart));
            commitMoves<<<(h_totalMoves + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(h_totalMoves, thrust::raw_pointer_cast(d_moveBuffer.data()), thrust::raw_pointer_cast(d_selectedFlag.data()), thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_partCount.data()), numPartitions);
            CUDA_CHECK(cudaEventRecord(kernelStop));
            CUDA_CHECK(cudaEventSynchronize(kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
            refinementKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());

            float currentEdgeCutKernelTime = 0.0f;
            float currentEdgeCutOverheadTime = 0.0f;
            int currentEdgeCut = computeEdgeCut(nvtxs, nedges, d_xadj, d_adjncy, d_adjwgt, d_partition, numPartitions, currentEdgeCutKernelTime, currentEdgeCutOverheadTime, kernelStart, kernelStop);
            edgeCutKernelTime += currentEdgeCutKernelTime;

            thrust::host_vector<int> h_partCount = d_partCount;
            float balanceScore = 0.0f;
            balanced = true;
            for (int q = 0; q < numPartitions; q++) {
                float diff = (float)h_partCount[q] - targetSize;
                balanceScore += diff * diff;
                if (abs(diff) > targetSize * maxImbalance) balanced = false;
            }
            balanceScore = sqrt(balanceScore / numPartitions);
            if (currentEdgeCut < bestEdgeCut || (currentEdgeCut <= bestEdgeCut * 1.01f && balanceScore < bestBalanceScore)) {
                bestEdgeCut = currentEdgeCut;
                bestBalanceScore = balanceScore;
                thrust::copy(d_partition.begin(), d_partition.end(), d_bestPartition.begin());
            }

            edgeCut = currentEdgeCut;
            if (h_totalMoves < nvtxs / 200 && thrust::count(d_selectedFlag.begin(), d_selectedFlag.begin() + h_totalMoves, true) < nvtxs / 400 && balanced) {
                break;
            }
        } else {
            break;
        }
    }

    thrust::copy(d_bestPartition.begin(), d_bestPartition.end(), d_partition.begin());
    float finalEdgeCutKernelTime = 0.0f;
    float finalEdgeCutOverheadTime = 0.0f;
    edgeCut = computeEdgeCut(nvtxs, nedges, d_xadj, d_adjncy, d_adjwgt, d_partition, numPartitions, finalEdgeCutKernelTime, finalEdgeCutOverheadTime, kernelStart, kernelStop);
    edgeCutKernelTime += finalEdgeCutKernelTime;

    eventOverheadStart = getTime();
    CUDA_CHECK(cudaFree(d_globalMoveCount));
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));
    refinementOverheadTime += getTime() - eventOverheadStart;

    refinementOverheadTime += getTime() - start - refinementKernelTime;
}

// Kernel to map partitions from coarse to fine graph during uncoarsening.
__global__ void kwayMapCoarseToFine(int* traceBack, int* partition, int* current_partition, int coarse_nvtxs, int fine_nvtxs, int numPartitions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= coarse_nvtxs) return;
    int coarse_part = partition[i];
    if (coarse_part >= numPartitions || coarse_part < 0) return;
    int v1 = traceBack[i * 2];
    int v2 = traceBack[i * 2 + 1];
    if (v1 >= 0 && v1 < fine_nvtxs) current_partition[v1] = coarse_part;
    if (v2 >= 0 && v2 < fine_nvtxs) current_partition[v2] = coarse_part;
}

// Function to check if the match array has converged between iterations.
bool checkMatchEquality(thrust::device_vector<int>& d_match, thrust::host_vector<int>& h_match_prev, int size) {
    thrust::host_vector<int> h_match_current(size);
    thrust::copy(d_match.begin(), d_match.begin() + size, h_match_current.begin());
    for (int i = 0; i < size; i++) {
        if (h_match_current[i] != h_match_prev[i]) return false;
    }
    return true;
}

// Function to read graph data from a file into an array.
void readFile(const char* fileName, int* array, int size) {
    FILE* file = fopen(fileName, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%d", &array[i]) != 1) {
            fprintf(stderr, "Error reading file %s at element %d\n", fileName, i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

// Function to convert CSR graph format to METIS format and run METIS for initial partitioning.
void convertCSRtoMETISandRun(const thrust::device_vector<int>& d_xadj, const thrust::device_vector<int>& d_adjncy, const thrust::device_vector<int>& d_adjwgt, int numPartitions, int& coarse_nvtxs, int& coarse_nedges, float& CSRToGraphConversionTime, float& GPMetisTime) {
    double start = getTime();
    thrust::host_vector<int> xadj = d_xadj;
    thrust::host_vector<int> adjncy = d_adjncy;
    thrust::host_vector<int> adjwgt = d_adjwgt;
    coarse_nvtxs = xadj.size() - 1;

    std::set<std::pair<int, int>> edge_set;
    std::vector<std::vector<int>> edge_weights(coarse_nvtxs, std::vector<int>(coarse_nvtxs, 0));

    for (int i = 0; i < coarse_nvtxs; i++) {
        int start = xadj[i];
        int end = (i + 1 < coarse_nvtxs) ? xadj[i + 1] : adjncy.size();
        for (int j = start; j < end; j++) {
            int neighbor = adjncy[j];
            int weight = adjwgt[j];
            if (neighbor >= 0 && neighbor < coarse_nvtxs && i != neighbor) {
                int u = std::min(i, neighbor);
                int v = std::max(i, neighbor);
                edge_set.insert({u, v});
                edge_weights[u][v] += weight;
            }
        }
    }
    coarse_nedges = edge_set.size();

    std::ofstream out("output.graph");
    if (!out.is_open()) {
        std::cerr << "Error: Cannot create output.graph\n";
        exit(EXIT_FAILURE);
    }
    out << coarse_nvtxs << " " << coarse_nedges << " 001\n";
    for (int i = 0; i < coarse_nvtxs; i++) {
        std::ostringstream line;
        std::set<int> written_neighbors;
        for (const auto& edge : edge_set) {
            int u = edge.first;
            int v = edge.second;
            if (u == i && written_neighbors.find(v) == written_neighbors.end()) {
                line << (v + 1) << " " << edge_weights[u][v] << " ";
                written_neighbors.insert(v);
            } else if (v == i && written_neighbors.find(u) == written_neighbors.end()) {
                line << (u + 1) << " " << edge_weights[u][v] << " ";
                written_neighbors.insert(u);
            }
        }
        std::string line_str = line.str();
        if (!line_str.empty() && line_str.back() == ' ') line_str.pop_back();
        out << line_str << "\n";
    }
    out.close();
    CSRToGraphConversionTime = getTime() - start;

    double metis_start = getTime();
    std::string cmd = "gpmetis -ptype=rb ./output.graph " + std::to_string(numPartitions);
    int ret = system(cmd.c_str());
    GPMetisTime = getTime() - metis_start;
    if (ret != 0) {
        std::cerr << "Error: gpmetis failed with return code " << ret << "\n";
        exit(EXIT_FAILURE);
    }
}

// Function to track maximum GPU memory usage during execution.
float updateMaxMemoryUsed(float& maxMemoryUsed, float& memoryCheckTime) {
    double start = getTime();
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    float used = (total - free) / (1024.0f * 1024.0f);
    maxMemoryUsed = std::max(maxMemoryUsed, used);
    memoryCheckTime += getTime() - start;
    return used;
}

// Main function: Orchestrates the graph partitioning process.
int main(int argc, char* argv[]) {
    // Check command-line arguments for directory, vertices, edges, and partitions.
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <directory> <nvtxs> <nedges> <num_partitions>\n";
        return 1;
    }

    const char* dir = argv[1];
    int nvtxs = std::atoi(argv[2]);
    int nedges = std::atoi(argv[3]);
    int numPartitions = std::atoi(argv[4]);

    // Validate input parameters.
    if (nvtxs <= 0 || nedges <= 0 || numPartitions <= 0) {
        std::cerr << "Invalid input: nvtxs, nedges, and numPartitions must be positive.\n";
        return 1;
    }

    // Set coarsening threshold: stop when vertices <= max(500, nvtxs/500).
    int target_vertices = std::max(500, nvtxs / 500);

    // Allocate host memory for graph data in CSR format.
    int* h_xadj = new int[nvtxs + 1];
    int* h_adjncy = new int[nedges];
    int* h_weight = new int[nedges];

    // Construct file paths for input graph data.
    char rowFile[256], columnFile[256], weightFile[256];
    snprintf(rowFile, sizeof(rowFile), "%s/row.txt", dir);
    snprintf(columnFile, sizeof(columnFile), "%s/column.txt", dir);
    snprintf(weightFile, sizeof(weightFile), "%s/weight.txt", dir);

    // Initialize variables for memory usage tracking.
    float memoryCheckTime = 0.0f;
    float maxMemoryUsed = 0.0f;
    updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);

    // Timing variables for profiling different phases.
    float FileReadingTime = 0.0f;
    float InitialHostToDeviceTransferTime = 0.0f;
    float CoarseningKernelTime = 0.0f;
    float CoarseningTransferTime = 0.0f;
    float CoarseningOverheadTime = 0.0f;
    float CSRToGraphConversionTime = 0.0f;
    float GPMetisTime = 0.0f;
    float PartitionFileReadingTime = 0.0f;
    float PartitionTransferTime = 0.0f;
    float UncoarseningTransferTime = 0.0f;
    float UncoarseningOverheadTime = 0.0f;
    float KwayMapCoarseToFineKernelTime = 0.0f;
    float RefinementKernelTime = 0.0f;
    float RefinementOverheadTime = 0.0f;
    float EdgeCutKernelTime = 0.0f;
    float EdgeCutOverheadTime = 0.0f;
    float FinalPartitionValueTransferTime = 0.0f;
    float PartitionFileWriteTime = 0.0f;

    float TotalDataTransferTime = 0.0f;
    float TotalPartitionTime = 0.0f;
    float TotalOverheadTime = 0.0f;

    double startTime;

    // Read graph data from files into host arrays.
    startTime = getTime();
    readFile(rowFile, h_xadj, nvtxs + 1);
    readFile(columnFile, h_adjncy, nedges);
    readFile(weightFile, h_weight, nedges);
    FileReadingTime = getTime() - startTime;

    // Transfer graph data from host to device.
    startTime = getTime();
    thrust::device_vector<int> d_xadj(nvtxs + 1);
    thrust::device_vector<int> d_adjncy(nedges);
    thrust::device_vector<int> d_adjwgt(nedges);
    thrust::copy(h_xadj, h_xadj + nvtxs + 1, d_xadj.begin());
    thrust::copy(h_adjncy, h_adjncy + nedges, d_adjncy.begin());
    thrust::copy(h_weight, h_weight + nedges, d_adjwgt.begin());
    InitialHostToDeviceTransferTime = getTime() - startTime;

    updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);

    // Initialize device vectors for coarsening and partitioning data.
    thrust::device_vector<int> d_match(nvtxs, -1);
    thrust::device_vector<int> d_tempweight(nvtxs, -1);
    thrust::device_vector<int> d_cmap(nvtxs, -1);
    thrust::device_vector<int> d_new_ids(nvtxs, -1);
    thrust::device_vector<int> d_label_map(nvtxs, -1);
    thrust::device_vector<int> d_traceBack(nvtxs * 2, -1);
    thrust::device_vector<int> d_partition(nvtxs, -1);
    thrust::device_vector<int> d_partCount(numPartitions, 0);

    int* d_coarsened_nvtxs;
    CUDA_CHECK(cudaMalloc(&d_coarsened_nvtxs, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_coarsened_nvtxs, 0, sizeof(int)));

    int blockSize = BLOCK_SIZE;
    int numBlocks = (nvtxs + blockSize - 1) / blockSize;
    int pass = 0;

    // Vectors to store history of coarsened graphs for uncoarsening.
    std::vector<int> vertex_count_history;
    std::vector<int> edge_count_history;
    std::vector<thrust::device_vector<int>> xadj_history;
    std::vector<thrust::device_vector<int>> adjncy_history;
    std::vector<thrust::device_vector<int>> weight_history;
    std::vector<thrust::device_vector<int>> traceBack_history;

    int coarsed_nvtx = nvtxs;
    int coarsed_edge = nedges;

    vertex_count_history.push_back(nvtxs);
    edge_count_history.push_back(nedges);
    xadj_history.push_back(d_xadj);
    adjncy_history.push_back(d_adjncy);
    weight_history.push_back(d_adjwgt);
    traceBack_history.push_back(thrust::device_vector<int>(nvtxs * 2, -1));

    cudaEvent_t start, stop;
    double eventOverheadStart = getTime();
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CoarseningOverheadTime += getTime() - eventOverheadStart;

    // Coarsening loop: Reduce graph size until it reaches the target number of vertices.
    while (coarsed_nvtx > target_vertices) {
        startTime = getTime();
        CUDA_CHECK(cudaMemset(d_coarsened_nvtxs, 0, sizeof(int)));
        thrust::fill(d_traceBack.begin(), d_traceBack.end(), -1);
        thrust::fill(d_cmap.begin(), d_cmap.end(), -1);
        thrust::fill(d_new_ids.begin(), d_new_ids.end(), -1);
        thrust::fill(d_label_map.begin(), d_label_map.end(), -1);
        CoarseningOverheadTime += getTime() - startTime;

        float kernelTimeMs;
        CUDA_CHECK(cudaEventRecord(start));
        resetMatch<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_match.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        matchVertices<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), thrust::raw_pointer_cast(d_match.data()), thrust::raw_pointer_cast(d_tempweight.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        int iteration = 0, max_iterations = 10;
        thrust::host_vector<int> h_match_prev(coarsed_nvtx, -1);
        bool converged = false;
        while (!converged && iteration < max_iterations) {
            startTime = getTime();
            thrust::copy(d_match.begin(), d_match.begin() + coarsed_nvtx, h_match_prev.begin());
            CoarseningTransferTime += getTime() - startTime;

            CUDA_CHECK(cudaEventRecord(start));
            rematchVertices <<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), thrust::raw_pointer_cast(d_match.data()), thrust::raw_pointer_cast(d_tempweight.data()), coarsed_nvtx);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
            CoarseningKernelTime += kernelTimeMs / 1000.0f;
            CUDA_CHECK(cudaDeviceSynchronize());
            converged = checkMatchEquality(d_match, h_match_prev, coarsed_nvtx);
            iteration++;
        }

        CUDA_CHECK(cudaEventRecord(start));
        mapVertices<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_match.data()), thrust::raw_pointer_cast(d_cmap.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        relabel_coarse_vertices_kernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_cmap.data()), thrust::raw_pointer_cast(d_new_ids.data()), thrust::raw_pointer_cast(d_label_map.data()), thrust::raw_pointer_cast(d_traceBack.data()), coarsed_nvtx, d_coarsened_nvtxs);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        startTime = getTime();
        int new_coarsed_nvtx;
        CUDA_CHECK(cudaMemcpy(&new_coarsed_nvtx, d_coarsened_nvtxs, sizeof(int), cudaMemcpyDeviceToHost));
        CoarseningTransferTime += getTime() - startTime;

        if (new_coarsed_nvtx <= 0 || new_coarsed_nvtx >= coarsed_nvtx) break;
        coarsed_nvtx = new_coarsed_nvtx;

        startTime = getTime();
        thrust::device_vector<int> d_uniqueCounts(coarsed_nvtx, 0);
        thrust::device_vector<int> d_row(coarsed_nvtx + 1, 0);
        thrust::device_vector<int> d_row_ptr(coarsed_nvtx + 1, 0);
        long long max_edges = static_cast<long long>(coarsed_edge) * 2;
        if (max_edges > INT_MAX) max_edges = INT_MAX;
        thrust::device_vector<int> d_col_idx(max_edges, 0);
        thrust::device_vector<int> d_values(max_edges, 0);
        CoarseningOverheadTime += getTime() - startTime;

        numBlocks = (coarsed_nvtx + blockSize - 1) / blockSize;
        
        CUDA_CHECK(cudaEventRecord(start));
        countUniqueEdges<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), thrust::raw_pointer_cast(d_new_ids.data()), thrust::raw_pointer_cast(d_match.data()), thrust::raw_pointer_cast(d_cmap.data()), thrust::raw_pointer_cast(d_uniqueCounts.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        prefix_sum<<<1, 1>>>(thrust::raw_pointer_cast(d_uniqueCounts.data()), thrust::raw_pointer_cast(d_row.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        fillEdgeArrays<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), thrust::raw_pointer_cast(d_new_ids.data()), thrust::raw_pointer_cast(d_match.data()), thrust::raw_pointer_cast(d_cmap.data()), thrust::raw_pointer_cast(d_row.data()), thrust::raw_pointer_cast(d_col_idx.data()), thrust::raw_pointer_cast(d_values.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        updateRowPtr<<<1, 1>>>(thrust::raw_pointer_cast(d_row.data()), thrust::raw_pointer_cast(d_row_ptr.data()), coarsed_nvtx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        CoarseningKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        startTime = getTime();
        thrust::host_vector<int> h_row = d_row;
        CoarseningTransferTime += getTime() - startTime;

        startTime = getTime();
        int new_coarsed_edge = h_row[coarsed_nvtx];
        if (new_coarsed_edge <= 0 || new_coarsed_edge > max_edges) {
            std::cerr << "Invalid new_coarsed_edge: " << new_coarsed_edge << "\n";
            exit(EXIT_FAILURE);
        }
        coarsed_edge = new_coarsed_edge;

        d_xadj.resize(coarsed_nvtx + 1);
        thrust::copy(d_row_ptr.begin(), d_row_ptr.begin() + coarsed_nvtx + 1, d_xadj.begin());
        d_adjncy.resize(new_coarsed_edge);
        thrust::copy(d_col_idx.begin(), d_col_idx.begin() + new_coarsed_edge, d_adjncy.begin());
        d_adjwgt.resize(new_coarsed_edge);
        thrust::copy(d_values.begin(), d_values.begin() + new_coarsed_edge, d_adjwgt.begin());

        d_match.resize(coarsed_nvtx);
        d_tempweight.resize(coarsed_nvtx);
        d_cmap.resize(coarsed_nvtx);
        d_new_ids.resize(coarsed_nvtx);
        d_label_map.resize(coarsed_nvtx);
        d_traceBack.resize(coarsed_nvtx * 2);
        CoarseningOverheadTime += getTime() - startTime;

        vertex_count_history.push_back(coarsed_nvtx);
        edge_count_history.push_back(coarsed_edge);
        xadj_history.push_back(d_xadj);
        adjncy_history.push_back(d_adjncy);
        weight_history.push_back(d_adjwgt);
        traceBack_history.push_back(d_traceBack);

        numBlocks = (coarsed_nvtx + blockSize - 1) / blockSize;
        pass++;
    }

    updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);

    // Perform initial partitioning on the coarsest graph using METIS.
    startTime = getTime();
    int coarse_nvtxs, coarse_nedges;
    convertCSRtoMETISandRun(xadj_history[pass], adjncy_history[pass], weight_history[pass], numPartitions, coarse_nvtxs, coarse_nedges, CSRToGraphConversionTime, GPMetisTime);

    // Read METIS partition output into host memory.
    startTime = getTime();
    std::string metis_partition_file = "output.graph.part." + std::to_string(numPartitions);
    std::ifstream fin(metis_partition_file);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open " << metis_partition_file << "\n";
        exit(EXIT_FAILURE);
    }
    thrust::host_vector<int> h_coarse_partition(coarse_nvtxs);
    for (int i = 0; i < coarse_nvtxs; i++) {
        fin >> h_coarse_partition[i];
        if (h_coarse_partition[i] < 0 || h_coarse_partition[i] >= numPartitions) h_coarse_partition[i] = 0;
    }
    fin.close();
    PartitionFileReadingTime = getTime() - startTime;

    // Transfer initial partition to device and initialize partition counts.
    startTime = getTime();
    d_partition.resize(coarse_nvtxs);
    thrust::copy(h_coarse_partition.begin(), h_coarse_partition.end(), d_partition.begin());
    thrust::fill(d_partCount.begin(), d_partCount.end(), 0);
    for (int i = 0; i < coarse_nvtxs; i++) {
        d_partCount[h_coarse_partition[i]]++;
    }
    PartitionTransferTime = getTime() - startTime;

    // Compute initial edge cut for the coarsest graph.
    startTime = getTime();
    float initialEdgeCutKernelTime = 0.0f;
    float initialEdgeCutOverheadTime = 0.0f;
    int initialEdgeCut = computeEdgeCut(coarse_nvtxs, coarse_nedges, thrust::raw_pointer_cast(xadj_history[pass].data()), thrust::raw_pointer_cast(adjncy_history[pass].data()), thrust::raw_pointer_cast(weight_history[pass].data()), d_partition, numPartitions, initialEdgeCutKernelTime, initialEdgeCutOverheadTime, start, stop);
    EdgeCutKernelTime += initialEdgeCutKernelTime;
    EdgeCutOverheadTime += initialEdgeCutOverheadTime;
    updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);

    // Refine the initial partition on the coarsest graph.
    startTime = getTime();
    float initialRefinementKernelTime = 0.0f;
    float initialRefinementOverheadTime = 0.0f;
    deterministicRefinement(coarse_nvtxs, coarse_nedges, thrust::raw_pointer_cast(xadj_history[pass].data()), thrust::raw_pointer_cast(adjncy_history[pass].data()), thrust::raw_pointer_cast(weight_history[pass].data()), d_partition, d_partCount, numPartitions, initialEdgeCut, EdgeCutKernelTime, initialRefinementKernelTime, initialRefinementOverheadTime);
    RefinementKernelTime += initialRefinementKernelTime;
    RefinementOverheadTime += initialRefinementOverheadTime;

    // Uncoarsening loop: Map partitions back to finer graphs and refine.
    for (int level = pass - 1; level >= 0; level--) {
        startTime = getTime();
        int current_nvtxs = vertex_count_history[level];
        int current_nedges = edge_count_history[level];
        int coarse_nvtxs_level = vertex_count_history[level + 1];

        d_xadj.resize(current_nvtxs + 1);
        d_adjncy.resize(current_nedges);
        d_adjwgt.resize(current_nedges);

        thrust::device_vector<int> d_coarse_partition(current_nvtxs, -1);
        UncoarseningOverheadTime += getTime() - startTime;

        startTime = getTime();
        thrust::copy(xadj_history[level].begin(), xadj_history[level].end(), d_xadj.begin());
        thrust::copy(adjncy_history[level].begin(), adjncy_history[level].end(), d_adjncy.begin());
        thrust::copy(weight_history[level].begin(), weight_history[level].end(), d_adjwgt.begin());
        UncoarseningTransferTime += getTime() - startTime;

        numBlocks = (coarse_nvtxs_level + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float kernelTimeMs;
        CUDA_CHECK(cudaEventRecord(start));
        kwayMapCoarseToFine<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(traceBack_history[level + 1].data()), thrust::raw_pointer_cast(d_partition.data()), thrust::raw_pointer_cast(d_coarse_partition.data()), coarse_nvtxs_level, current_nvtxs, numPartitions);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
        KwayMapCoarseToFineKernelTime += kernelTimeMs / 1000.0f;
        CUDA_CHECK(cudaDeviceSynchronize());

        startTime = getTime();
        d_partition.resize(current_nvtxs);
        UncoarseningOverheadTime += getTime() - startTime;

        startTime = getTime();
        thrust::copy(d_coarse_partition.begin(), d_coarse_partition.end(), d_partition.begin());
        UncoarseningTransferTime += getTime() - startTime;

        startTime = getTime();
        thrust::fill(d_partCount.begin(), d_partCount.end(), 0);
        thrust::host_vector<int> h_temp_partition = d_partition;
        for (int i = 0; i < current_nvtxs; i++) {
            if (h_temp_partition[i] >= 0 && h_temp_partition[i] < numPartitions) {
                d_partCount[h_temp_partition[i]]++;
            } else {
                h_temp_partition[i] = 0;
                d_partCount[0]++;
            }
        }
        UncoarseningOverheadTime += getTime() - startTime;

        startTime = getTime();
        thrust::copy(h_temp_partition.begin(), h_temp_partition.end(), d_partition.begin());
        UncoarseningTransferTime += getTime() - startTime;

        startTime = getTime();
        float currentEdgeCutKernelTime = 0.0f;
        float currentEdgeCutOverheadTime = 0.0f;
        int edgeCut = computeEdgeCut(current_nvtxs, current_nedges, thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), d_partition, numPartitions, currentEdgeCutKernelTime, currentEdgeCutOverheadTime, start, stop);
        EdgeCutKernelTime += currentEdgeCutKernelTime;
        EdgeCutOverheadTime += currentEdgeCutOverheadTime;

        startTime = getTime();
        float currentRefinementKernelTime = 0.0f;
        float currentRefinementOverheadTime = 0.0f;
        deterministicRefinement(current_nvtxs, current_nedges, thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), d_partition, d_partCount, numPartitions, edgeCut, EdgeCutKernelTime, currentRefinementKernelTime, currentRefinementOverheadTime);
        RefinementKernelTime += currentRefinementKernelTime;
        RefinementOverheadTime += currentRefinementOverheadTime;

        updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);
    }

    eventOverheadStart = getTime();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    UncoarseningOverheadTime += getTime() - eventOverheadStart;

    // Compute final edge cut and transfer partition to host.
    startTime = getTime();
    float finalEdgeCutKernelTime = 0.0f;
    float finalEdgeCutOverheadTime = 0.0f;
    int finalEdgeCut = computeEdgeCut(nvtxs, nedges, thrust::raw_pointer_cast(d_xadj.data()), thrust::raw_pointer_cast(d_adjncy.data()), thrust::raw_pointer_cast(d_adjwgt.data()), d_partition, numPartitions, finalEdgeCutKernelTime, finalEdgeCutOverheadTime, start, stop);
    EdgeCutKernelTime += finalEdgeCutKernelTime;
    EdgeCutOverheadTime += finalEdgeCutOverheadTime;
    thrust::host_vector<int> h_final_partition(nvtxs);
    thrust::copy(d_partition.begin(), d_partition.end(), h_final_partition.begin());
    FinalPartitionValueTransferTime = getTime() - startTime - finalEdgeCutKernelTime;

    updateMaxMemoryUsed(maxMemoryUsed, memoryCheckTime);

    // Write final partition to output file.
    startTime = getTime();
    std::ofstream output_partition_file("./finalpartition.txt");
    for (int i = 0; i < nvtxs; i++) {
        output_partition_file << h_final_partition[i];
        if (i < nvtxs - 1) output_partition_file << ", ";
        else output_partition_file << "\n";
    }
    output_partition_file.close();
    PartitionFileWriteTime = getTime() - startTime;

    // Compute total times for data transfer, partitioning, and overhead.
    TotalDataTransferTime = InitialHostToDeviceTransferTime + CoarseningTransferTime + PartitionTransferTime + UncoarseningTransferTime + FinalPartitionValueTransferTime + memoryCheckTime;
    TotalPartitionTime = CoarseningKernelTime + KwayMapCoarseToFineKernelTime + RefinementKernelTime ;
    TotalOverheadTime = FileReadingTime + CoarseningOverheadTime + CSRToGraphConversionTime + GPMetisTime + EdgeCutOverheadTime + PartitionFileReadingTime + PartitionFileWriteTime + UncoarseningOverheadTime + RefinementOverheadTime + EdgeCutKernelTime;

    // Output graph details, partitioning results, and timing breakdown.
    std::cout << "Graph Detail ******************************************************\n";
    std::cout << std::left << std::setw(35) << "Number of Vertices:" << std::right << std::setw(10) << nvtxs << "\n";
    std::cout << std::left << std::setw(35) << "Number of Edges:" << std::right << std::setw(10) << nedges << "\n";
    std::cout << std::left << std::setw(35) << "Number of Partitions:" << std::right << std::setw(10) << numPartitions << "\n\n";

    std::cout << "K-way GPU with Deterministic Refinement Details *******************\n";
    std::cout << std::left << std::setw(35) << "Edge Cut:" << std::right << std::setw(10) << finalEdgeCut << "\n";
    std::cout << "Partition Sizes:\n";
    thrust::host_vector<int> h_partCount = d_partCount;
    for (int q = 0; q < numPartitions; q++) {
        std::cout << "  Partition " << q << ":" << std::right << std::setw(10) << h_partCount[q] << " vertices\n";
    }
    std::cout << std::left << std::setw(35) << "Balance Constraint:" << std::right << std::setw(10) << std::fixed << std::setprecision(1) << (MAX_IMBALANCE * 100.0f) << "%\n";
    std::cout << std::left << std::setw(35) << "Coarsening Threshold:" << std::right << std::setw(10) << target_vertices << " vertices\n";
    std::cout << std::left << std::setw(35) << "Output File:" << std::right << std::setw(10) << "./finalpartition.txt" << "\n\n";

    std::cout << "Timing Breakdown **************************************************\n";
    std::cout << "Total Data Transfer Time:\n";
    std::cout << std::left << std::setw(35) << "  - File Reading Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << FileReadingTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Initial Host to Device Transfer:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << InitialHostToDeviceTransferTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Coarsening Transfer Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << CoarseningTransferTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Partition Transfer Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << PartitionTransferTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Uncoarsening Transfer Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << UncoarseningTransferTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Final Partition Transfer Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << FinalPartitionValueTransferTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Memory Check Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << memoryCheckTime << "s\n";
    std::cout << std::left << std::setw(35) << "Total Data Transfer Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << TotalDataTransferTime << "s\n\n";
    std::cout << "Total Partition Time:\n";
    std::cout << std::left << std::setw(35) << "  - Coarsening Kernel Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << CoarseningKernelTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - K-way Map Coarse to Fine Kernel:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << KwayMapCoarseToFineKernelTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Refinement Kernel Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << RefinementKernelTime << "s\n";
    
    std::cout << std::left << std::setw(35) << "Total Partition Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << TotalPartitionTime << "s\n\n";

    std::cout << "Total Overhead Time:\n";
    std::cout << std::left << std::setw(35) << "  - File Reading Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << FileReadingTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Coarsening Overhead Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << CoarseningOverheadTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - CSR to Graph Conversion Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << CSRToGraphConversionTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - GPMetis Execution Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << GPMetisTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Edge Cut Overhead Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << EdgeCutOverheadTime + EdgeCutKernelTime << "s\n";
  
    std::cout << std::left << std::setw(35) << "  - Partition File Reading Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << PartitionFileReadingTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Partition File Write Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << PartitionFileWriteTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Uncoarsening Overhead Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << UncoarseningOverheadTime << "s\n";
    std::cout << std::left << std::setw(35) << "  - Refinement Overhead Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << RefinementOverheadTime << "s\n";
    std::cout << std::left << std::setw(35) << "Total Overhead Time:" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << TotalOverheadTime << "s\n\n";

    std::cout << "Memory Information ************************************************\n";
    std::cout << std::left << std::setw(35) << "Maximum Memory Used:" << std::right << std::setw(10) << std::fixed << std::setprecision(2) << maxMemoryUsed << " MB\n\n";

    std::cout << "************************** Amitesh Singh **************************\n";
    std::cout << "**************************** Rupam Roy ****************************\n";
    std::cout << "************** Indian Institute of Technology Bhilai **************\n";

    // Clean up allocated memory.
    delete[] h_xadj;
    delete[] h_adjncy;
    delete[] h_weight;
    CUDA_CHECK(cudaFree(d_coarsened_nvtxs));

    return 0;
}