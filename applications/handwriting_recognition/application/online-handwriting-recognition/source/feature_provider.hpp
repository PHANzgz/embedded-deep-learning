#ifndef _FEATURE_PROVIDER_H_
#define _FEATURE_PROVIDER_H_

#include <stdint.h>
#include <stddef.h>

/**
*   DataProvider class - Stores the accelerometer data in a ring buffer and
*   provides the latest data to the Edge Impulse SDK
*/
class DataProvider {

    public:
        /**
        *   Class constructor
        *   @param n_timesteps number of timesteps that will be fed into the neural network
        *   @param n_features number of features the sensor data has. Three for accelerometer data.
        *   @param data_buffer the user is in charge of allocating memory for the ring buffer
        *   @param buffer_size ring buffer size, must be big enough to ensure data is not overwritten
        *       when retrieving it
        *   @param gain scale factor for the sensor raw data
        */
        DataProvider(int n_timesteps, 
                     int n_features, 
                     float* data_buffer, 
                     int buffer_size, 
                     float gain=1.);

        /**
        *   Class destructor
        */
        ~DataProvider();

        /**
        *   Insert a new sample into the ring buffer. For an accelerometer there are three features,
        *   one for each axis.
        *   @param ax acceleration on the x axis
        *   @param ay acceleration on the y axis
        *   @param az acceleration on the z axis
        */
        void pushData(float ax, float ay, float az);

        /**
        *   This function must be set as a callback for the get_data function of the input signal_t of
        *   the neural network. The signature matches with the Edge Impulse SDK one.
        *   Fills a float array with the latest accelerometer data correctly ordered.
        */
        int retrieveData(size_t offset, size_t length, float *out_ptr);

        /**
        * Utility to get the expected NN input element count
        */
        size_t getInputSize();

    private:
        float* data_buffer_;
        int n_timesteps_;
        int n_features_;
        int buffer_size_;
        float gain_;
        int head;
        inline int wrapIndex(int ix, int array_size);


};


#endif // _FEATURE_PROVIDER_H_