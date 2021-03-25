#include "output_handler.hpp"

#include "utils_ohw.hpp"
#include <stdint.h>

OutputHandler::OutputHandler(unsigned int n_labels, 
                             uint8_t window_size, 
                             float threshold_probability,
                             uint8_t suppresion_window_size)
    : n_labels_(n_labels),
      window_size_(window_size),
      threshold_probability_(threshold_probability),
      suppresion_window_size_(suppresion_window_size),
      got_detection(false),
      current_row_ix(0){

        // Allocate memory for a 2D array with window_size rows and n_labels columns
        // len = size of array of pointers for every row + memory for every row
        int len = sizeof(float *) * window_size + sizeof(float) * n_labels * window_size;
        
        float **arr = (float **) malloc(len);
        float *tmp = (float *) arr + window_size;

        // Initialize pointers to each row
        for(int i = 0; i < window_size; i++){
            arr[i] = (tmp + n_labels * i);
        }

        latest_predictions_buffer = arr;

        // Allocate memory for the scores moving average results
        scores_mov_avg = (float *) calloc(n_labels, sizeof(float));

    }


OutputHandler::~OutputHandler(){
    free(latest_predictions_buffer);
    free(scores_mov_avg);
}


void OutputHandler::insertNewPrediction(float *prediction){

    // Copy new prediction to the oldest entry of the predictions buffer
    memcpy(latest_predictions_buffer[current_row_ix], prediction, n_labels_ * sizeof(float));

    current_row_ix = (current_row_ix + 1) % window_size_;
}


void OutputHandler::processLatestPredictions(){

    // Mean across axis 0 (it would be enough with the sum, but this is nice for debugging and intuition)
    for (size_t col = 0; col < n_labels_; col++){

        float sum = 0.0f;

        for (size_t row = 0; row < window_size_; row++){
            sum += latest_predictions_buffer[row][col];
        }

        scores_mov_avg[col] = sum / window_size_;
    }

    // Max and argmax across the mean vector
    maximum_score = -1.0f; // Mean vector is positive
    for(size_t i = 0; i < n_labels_; i++){

        if(scores_mov_avg[i] > maximum_score){
            maximum_score = scores_mov_avg[i];
            arg_maximum_score = i;
        }
    }

}

void OutputHandler::Handle(float *prediction, USBSerial& pc, NNOutputHandlerService& ble_service){

    static uint8_t n_times_suppresed = 0;

    // Add and process new prediction
    insertNewPrediction(prediction);
    processLatestPredictions();

    // Always send scores over BLE (NON-PORTABLE)
    uint8_t avg_scores[n_labels_];
    for(size_t i=0; i < n_labels_; i++){
        avg_scores[i] = (uint8_t) (scores_mov_avg[i] * 255);
    }
    ble_service.set_new_scores(avg_scores);

    // Continue or suppress output
    if (got_detection){ // && n_times_suppresed < suppresion_window_size_ ){
        if (n_times_suppresed < suppresion_window_size_){
            n_times_suppresed++;
            return;
        } else {
            got_detection = false;
        }
    }

    // Detect a robust prediction
    // Only when the prediction is not 'noise'
    if ( (maximum_score > threshold_probability_) && (arg_maximum_score != (n_labels_ - 1)) ){
        got_detection = true;
        n_times_suppresed = 0;
    }

    /* Non-portable user defined output handle */

    // Print prediction to user and send it over BLE
    if (got_detection){
        pc.printf("You wrote the letter %c, with a confidence of %2.2f \r\n", 
                  ('A' + arg_maximum_score), maximum_score);
        ble_service.set_new_prediction(arg_maximum_score);
    }


}

