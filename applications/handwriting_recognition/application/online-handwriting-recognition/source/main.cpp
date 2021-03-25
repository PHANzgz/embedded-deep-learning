/*
 *  Online handwriting recognition  
 *
 *  Author: Enrique Phan
 *  Github: PHANzgz
 *
 *  Description:
 *  With an accelerometer as a pen cap, data is sampled continuously and 
 *  filtered. The data is fed to the convolutional neural network which 
 *  also runs continously with the latest 2 seconds of accelerometer data. 
 *  Once a letter is written on paper and a prediction is made, the 
 *  information is sent over BLE. 
 *
 *  Preprocessing takes approx 10ms.
 *  Inference takes approx 62ms.
 *  
 *  Date: March 2021
 */

/* User includes ----------------------------------------------------------*/
#include "mbed.h"
#include "rtos.h"
//#include <chrono>

#include "ei_run_classifier.h"
#include "numpy.hpp"

#include "feature_provider.hpp"
#include "output_handler.hpp"
#include "utils_ohw.hpp"
// #include "test_data.h" // Used for unit testing

// BLE related
#include "events/EventQueue.h"
#include "ble/BLE.h"
#include "nn_output_handler_service.hpp"

// Accelerometer driver
#include "LSM9DS1.h"


/* User define ------------------------------------------------------------*/
#define N_TIMESTEPS                         200
#define N_FEATURES                          3
#define INPUT_BUFFER_SIZE                   N_TIMESTEPS*N_FEATURES
#define OVERHEAD_TIMESTEPS                  10 // Ensures data is not overwritten in the ring buffer when reading
#define RING_BUFFER_SIZE                    (N_TIMESTEPS+OVERHEAD_TIMESTEPS)*N_FEATURES
#define LSM9DS1_ACCEL_SCALE                 LSM9DS1::A_SCALE_4G
#define LSM9DS1_ACCEL_SAMPLE_RATE           LSM9DS1::A_ODR_119
#define PIN_I2C_PULLUP                      P1_0
#define GET_DATA_TASK_PERIOD_MS             10ms
#define INFERENCE_TASK_PERIOD_MS            150ms
#define THRESHOLD_PROBABILITY               0.75
#define OUTPUT_HANDLER_WINDOW_SIZE          7
#define SUPPRESSION_WINDOW_SIZE             8


// ROM static and global variables

// RAM static and global variables
static float data_buffer[RING_BUFFER_SIZE] = {0};   // Ring buffer where acc data will be saved

// We use a gain of 10 to mimic the data acquisition format
DataProvider data_provider(N_TIMESTEPS, N_FEATURES, data_buffer, RING_BUFFER_SIZE, 10);

// Window size and thresh probability must be tweaked for good performance
OutputHandler output_handler(EI_CLASSIFIER_LABEL_COUNT, 
                             OUTPUT_HANDLER_WINDOW_SIZE, 
                             THRESHOLD_PROBABILITY,
                             SUPPRESSION_WINDOW_SIZE);

DigitalOut      led1(LED1);
DigitalOut      sen_pwr(VDD_ENV, 1);
DigitalOut      i2c_pullup(PIN_I2C_PULLUP, 1);
I2C             i2c(I2C_SDA1, I2C_SCL1);
LSM9DS1         *imu = NULL;
Timer           t;

// BLE related
static BLE &ble_instance = BLE::Instance();
static events::EventQueue event_queue;
static NNOutputHandlerService output_handler_service;


// Function prototypes
void get_accelerometer_data_task();
void inference_task();
void ble_task();


int main() {

    // Task creation
    Thread get_accelerometer_data_thread(osPriorityAboveNormal1, 4096);
    Thread ble_thread(osPriorityAboveNormal, 4096);
    Thread inference_thread(osPriorityNormal, 4096);

    // Start threads
    get_accelerometer_data_thread.start(get_accelerometer_data_task);
    inference_thread.start(inference_task);
    ble_thread.start(ble_task);

    // Idle task, useful for debugging
    while (true) {
        led1 = !led1;
        ThisThread::sleep_for(500ms);
    }
    // Program should never get here
    NVIC_SystemReset(); 
}


void get_accelerometer_data_task(){

    // Sensor initialization
    i2c_pullup = 1;
    sen_pwr = 1;
    ThisThread::sleep_for(200ms);

    imu = new LSM9DS1(I2C_SDA1, I2C_SCL1);
    uint16_t reg = imu->begin(
            LSM9DS1::G_SCALE_245DPS,   // Arbitratry (unused)
            LSM9DS1_ACCEL_SCALE,
            LSM9DS1::M_SCALE_4GS,      // Arbitrary (unused)
            LSM9DS1::G_POWER_DOWN,     // Gyro OFF
            LSM9DS1_ACCEL_SAMPLE_RATE,
            LSM9DS1::M_ODR_0625        // Lowest available (unused)
            );

    imu->calibration();
    if (reg == 0x683d) pc.printf("Connected to LSM9DS1\r\n");

    Kernel::Clock::time_point next = Kernel::Clock::now();
    while (1){

        //pc.printf("Executing get_accelerometer_data_task at time %llu \r\n", Kernel::Clock::now().time_since_epoch().count() );

        // Read acceleration data
        imu->readAccel();
        data_provider.pushData(imu->ax, imu->ay, imu->az);

        // Debugging info
        //pc.printf("ax is %+2.4f, ay is %+2.4f, az is %+2.4f \r\n", imu->ax, imu->ay, imu->az);

        next += GET_DATA_TASK_PERIOD_MS;
        ThisThread::sleep_until(next);
    }
}



void inference_task(){

    // Edge impulse init
    ThisThread::sleep_for(1s); // Lazy but effective way to let the BLE task to initialize
    pc.printf("Online handwriting recognition inferencing (Mbed)\r\n");

    size_t input_element_count = data_provider.getInputSize();

    // Sanity check (not really neede once tested though)
    if (input_element_count != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        pc.printf("The size of your 'input_buffer' array is not correct. Expected %d items, but had %u\r\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, input_element_count);
    }

    // Signal data
    signal_t features_signal;
    features_signal.total_length = input_element_count;
    features_signal.get_data = callback(&data_provider, &DataProvider::retrieveData);

    // Result data
    ei_impulse_result_t result = { { 0 } };

    // Parsed result data
    float prediction[EI_CLASSIFIER_LABEL_COUNT];

    Kernel::Clock::time_point next = Kernel::Clock::now();
    while(1){

        //pc.printf("Executing inference_task at time %llu \r\n", Kernel::Clock::now().time_since_epoch().count());

        //t.start();
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
        //t.stop();
        if(res != EI_IMPULSE_OK) pc.printf("Error running classifier \r\n");

        //pc.printf("Inference took %llu milliseconds\r\n", 
        //         std::chrono::duration_cast<std::chrono::milliseconds>(t.elapsed_time()).count());
        //t.reset();

        for(size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++){
            prediction[i] = result.classification[i].value;
        }

        output_handler.Handle(prediction, pc, output_handler_service);

        //pc.printf("Timing summary: (DSP: %d ms., Classification: %d ms.): \r\n",
        //    result.timing.dsp, result.timing.classification);
        // Currently -> DSP: 10ms   Classification: 62ms.

        next += INFERENCE_TASK_PERIOD_MS;
        ThisThread::sleep_until(next);

    }
}

void ble_task(){

    // this process will handle basic ble setup and advertising
    static GattServerProcess ble_process(event_queue, ble_instance);

    // start process
    ble_process.on_init(callback(&output_handler_service, 
                                 &NNOutputHandlerService::start));
    ble_process.start();

}