#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "load_save.h"
#include "blur_ops.h"

using namespace std;

size_t numRows, numCols;

uchar4* load_image_in_GPU(string filename) {
	// Load the image into main memory
	uchar4 *h_image = NULL, *d_in = NULL;
	loadImageRGBA(filename, &h_image, &numRows, &numCols);
 	// Allocate memory to the GPU
	cudaMalloc((void **) &d_in, numRows * numCols * sizeof(uchar4));
	cudaMemcpy(d_in, h_image, numRows * numCols * sizeof(uchar4), cudaMemcpyHostToDevice);
	// No need to keep this image in RAM now.
	delete h_image;
	return d_in;
}

int hex_to_int(string hexStr) {
	int i;
	stringstream ss;
    	ss << std::hex << hexStr;
	ss >> i;
	return i;
}

uchar4 hex_to_uchar4_color(string& color) {
	int r = hex_to_int(color.substr(0, 2));
	int g = hex_to_int(color.substr(2, 2));
	int b = hex_to_int(color.substr(4, 2));
	return make_uchar4(r, g, b, 255);
}

int main(int argc, char **argv) {

	string input_file = "gpu.jpg";
	string output_file = "d_out.jpg";
	uchar4 *d_in = load_image_in_GPU(input_file);
	uchar4 *h_out = NULL;

	// Performing the required operation
	int amount = 20;
	if(amount % 2 == 0)
		amount++;
	h_out = blur_ops(d_in, numRows, numCols, amount);

	cudaFree(d_in);
	if(h_out != NULL)
		saveImageRGBA(h_out, output_file, numRows, numCols);

}
