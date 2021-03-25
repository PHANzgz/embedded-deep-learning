#include "feature_provider.hpp"

DataProvider::DataProvider(int n_timesteps, 
                           int n_features, 
                           float* data_buffer, 
                           int buffer_size, 
                           float gain)
    : data_buffer_(data_buffer),
      n_timesteps_(n_timesteps),
      n_features_(n_features),
      buffer_size_(buffer_size),
      gain_(gain){

    head = 0;

}

DataProvider::~DataProvider() {} 


void DataProvider::pushData(float ax, float ay, float az){

    // Store data in ring buffer
    data_buffer_[head] = gain_ * ax;
    data_buffer_[head + 1] = gain_ * ay;
    data_buffer_[head + 2] = gain_ * az;

    // head points to the NEXT data index
    head = (head + 3) % buffer_size_;

}


int DataProvider::retrieveData(size_t offset, size_t length, float *out_ptr){

    // Compute index that points to the beginning of the latest 2s + offset
    int start_ix = wrapIndex(head - n_timesteps_*n_features_ + offset , buffer_size_);

    for(size_t i=0; i < length; i++){

        //int ring_buffer_ix = wrapIndex(start_ix + i, buffer_size_);
        int ring_buffer_ix = (start_ix + i) % buffer_size_; // Faster
        
        out_ptr[i] = data_buffer_[ring_buffer_ix];

    }

    return 0;
}


size_t DataProvider::getInputSize(){
    return n_timesteps_ * n_features_;
}


inline int DataProvider::wrapIndex(int ix, int array_size){
    return ((ix % array_size) + array_size) % array_size;
}