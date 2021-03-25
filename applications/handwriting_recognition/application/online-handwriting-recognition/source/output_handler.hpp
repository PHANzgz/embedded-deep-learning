#ifndef _OUTPUT_HANDLER_
#define _OUTPUT_HANDLER_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Includes for the non-portable output handle
#ifdef __MBED__
#include "mbed.h"
#include "USBSerial.h"
#include "events/EventQueue.h"
#include "ble/BLE.h"
#include "nn_output_handler_service.hpp"
#endif


/**
*   OutputHandler class - Used to process the predictions of the neural network.
*   Its main purpose is to average over the latest predictions and produce the
*   desired output.
*
*   Note: For big window sizes, an exponential moving average might be a better
*         choice as it does not require a buffer to store the latest predictions.
*/
class OutputHandler{
    public:

        /**
        *   Class constructor
        *   @param n_labels number of labels of the neural network
        *   @param window_size number of predictions to average over, which is closely related to
        *       the amount of time to average over
        *   @param threshold_probability threshold value to consider a positive prediction. This
        *       value is compared once the mean scores are computed. It should be 0 <= x <= 1 .
        *   @param suppresion_window_size number of predictions to ignore after a succesful detection
        */
        OutputHandler(unsigned int n_labels, 
                      uint8_t window_size, 
                      float threshold_probability,
                      uint8_t suppresion_window_size);

        /**
        *   Class destructor
        *   Frees the memory allocated for the different buffers
        */
        ~OutputHandler();

        /**
        *   Insert a new prediction in the latest_predictions_buffer, overwriting the oldest one
        *   @param prediction array containing the output probabilities for each label produced by
        *       the model
        */
        void insertNewPrediction(float *prediction);

        /**
        *   Calculate the mean across the latest_predictions_buffer for each label(along axis 0).
        *   Then obtain the maximum_score of the mean scores and its index(arg_maximum_score).
        */
        void processLatestPredictions();

        /**
        *   This is the only non-portable function, which should be adapted for each target
        *   and application. In this case it prints out to the user the prediction information
        *   @param prediction array containing the output probabilities for each label produced by
        *       the model
        *   @param pc USBSerial instance to print through UART
        */
        void Handle(float *prediction, USBSerial& pc, NNOutputHandlerService& ble_service);


    private:
        unsigned int n_labels_;             // Number of labels in the model
        uint8_t window_size_;               // Number of predictions to average over(i.e. amount of time to average over)
        float threshold_probability_;       // Threshold value to consider a positive prediction
        uint8_t suppresion_window_size_;    // Number of predictions to ignore after a succesful detection
        bool got_detection;                 // Flag to indicate if a new detection was found
        float** latest_predictions_buffer;  // 2D Array where the latest predictions are stored
        uint8_t current_row_ix;             // Current row index for circular indexing
        float * scores_mov_avg;             // Probability mean across each prediction for every label (axis 0)
        float maximum_score;                // Maximum score of the mean scores
        uint8_t arg_maximum_score;          // Index of the maximum score of the mean scores
};

#endif // _OUTPUT_HANDLER_