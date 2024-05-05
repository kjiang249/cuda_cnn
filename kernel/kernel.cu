// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void convolution_layer(double input[100][100], double conv_filters[10][5][5], double outputs[10][20][20]) {
    // Calculate the row and column index of the output element to work on
    int col = threadIdx.x;
    int row = threadIdx.y;
    int filterIndex = blockIdx.z; // Assuming each filter is processed in a different block in the z dimension

    if (row >= 20 || col >= 20 || filterIndex >= 10) return;

    double sum = 0.0;

    // Convolution operation
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            int inputRow = row * 5 + i;
            int inputCol = col * 5 + j;
            double inputValue = input[inputRow][inputCol];
            double filterValue = conv_filters[filterIndex][i][j];
            sum += inputValue * filterValue;
        }
    }

    // Write the sum to the output
    outputs[filterIndex][row][col] = sum;
}

extern "C" __global__ void relu_layer(double data[10][20][20]) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // Each block in Z handles one depth layer

    if (row >= 20 || col >= 20 || z >= 10) return;

    if (data[z][row][col] < 0.0) {
        data[z][row][col] = 0.0;
    }
    
}

extern "C" __global__ void output_layer(double *input, double weights[10][4000], double *output) {
    int idx = threadIdx.x;
    if (idx < 10) {
        double sum = 0.0;
        for (int i = 0; i < 4000; i++) {
            sum += input[i] * weights[idx][i];
        }
        output[idx] = sum;
    }
}

extern "C" __global__ void elementwise_multiply(double *input1, double weights[10][4000], double output[10][4000]) {
    int z = blockIdx.z;
    // if (threadIdx.x >= 500) return;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < 4000) {
        output[z][idx] = input1[idx] * weights[z][idx];
    }
}

extern "C" __global__ void partial_add(double input1[10][4000], double output[10][200]) {
    int z = blockIdx.z; // Filter index
    int segmentIndex = threadIdx.x; // Index for each segment in the 200-segment array

    if(segmentIndex >= 200) return; // Guard against excess threads
    
    double sum = 0.0;
    // Calculate the starting index for the segment
    int startIdx = segmentIndex * 20; 
    // Sum over the segment
    for (int i = 0; i < 20; i++) {
        sum += input1[z][startIdx + i];
    }
    output[z][segmentIndex] = sum;
}

extern "C" __global__ void final_add(double input1[10][200], double output[10]) {
    int z = threadIdx.x;
    for(int i = 0; i < 200;i++) {
        output[z] += input1[z][i];
    }
}


// extern "C" __global__ void partial_add(double input1[10][4000], double output[10][200]) {
//     int z = blockIdx.z;

//     int idx = threadIdx.x;
    
//     if(idx >= 200) return;
    
//     // for (int i = idx; i < idx + 20; i++) {
//     //     output[z][idx] += input1[z][i];
//     // }
//     double sum = 0.0;
//     for (int i = 0; i < 20; i++) {
//         sum += input1[z][idx * 20 + i]; // Notice the corrected indexing
//     }
//     output[z][idx] = sum;
// }

// extern "C" __global__ void final_add(double input1[10][200], double output[10]) {
//     int z = blockIdx.z;
//     int idx = threadIdx.x;

//     if(idx >= 200) return;
    
//     output[z] += input1[z][idx];
    
// }


// extern "C" __global__ void test_add(double input1[10][4000], double output[10]) {
//     int z = blockIdx.z;
//     for(int i = 0; i < 4000;i++) {
//         output[z] += input1[z][i];
//     }
// }