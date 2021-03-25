#include "utils_ohw.hpp"

USBSerial pc;

void print_array_float(float *arr, size_t len){

    pc.printf("{");
    for (size_t i=0; i < len; i++) {
        if (i != 0) pc.printf(", ");
        pc.printf("%2.4f", arr[i]);
    }
    pc.printf("}\r\n");

}