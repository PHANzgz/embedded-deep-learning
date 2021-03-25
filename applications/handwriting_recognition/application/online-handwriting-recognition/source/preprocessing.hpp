#ifndef _PREPROCESSING_H_
#define _PREPROCESSING_H_

/* User includes ----------------------------------------------------------*/
#include "numpy.hpp"
#include "edge-impulse-sdk/dsp/spectral/spectral.hpp"
#include "ei_model_types.h"
//#include "utils_ohw.hpp" // Just for debugging

/* Private define ------------------------------------------------------------*/


// Namespace
using namespace ei;

/* User typedef -----------------------------------------------------------*/
typedef struct {
    uint16_t implementation_version;
    int axes;
    float scale_axes;
    bool center_data;
    int filter_order;
    float cutoff;
} ei_dsp_config_custom_t;



int extract_custom_block_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float f){
    
    ei_dsp_config_custom_t config = *((ei_dsp_config_custom_t*)config_ptr);

    //pc.printf("config.cutoff is %.2f, config.filter_order is %d,"
    //           "config.center_data is %d\r\n", config.cutoff, config.filter_order, config.center_data);

    int ret;


    // input matrix from the raw signal, shape=(n_timesteps, n_features)
    matrix_t input_matrix(signal->total_length / config.axes, config.axes);
    signal->get_data(0, signal->total_length, input_matrix.buffer);
    
    // Mean matrix, shape=(n_features, 1)
    matrix_t mean_matrix(config.axes, 1);

    // Compute the mean along time axis (axis 0)
    ret = numpy::mean_axis0(&input_matrix, &mean_matrix);
    if (ret != EIDSP_OK) {
        EIDSP_ERR(ret);
    }

    // Transpose for future operations
    ret = numpy::transpose(&input_matrix);
    if (ret != EIDSP_OK) {
        EIDSP_ERR(ret);
    }

    // Subtract the mean (axis 0, i.e features)
    ret = numpy::subtract(&input_matrix, &mean_matrix);
    if (ret != EIDSP_OK) {
        EIDSP_ERR(ret);
    }

    // Apply low pass filter
    spectral::processing::butterworth_lowpass_filter(&input_matrix, f, 
                                                 config.cutoff, config.filter_order);

    // Transpose again, since the NN expects shape=(n_timesteps, n_features)
    ret = numpy::transpose(&input_matrix);
    if (ret != EIDSP_OK) {
        EIDSP_ERR(ret);
    }

    //pc.printf("Done\r\n");
    //pc.printf("n_rows is %u and n_cols is %u\r\n", (unsigned int)input_matrix.rows, (unsigned int)input_matrix.cols);

    memcpy(output_matrix->buffer, input_matrix.buffer, signal->total_length * sizeof(float));
    //print_array_float(output_matrix->buffer, signal->total_length);

    return EIDSP_OK;
}

#endif // _PREPROCESSING_H_