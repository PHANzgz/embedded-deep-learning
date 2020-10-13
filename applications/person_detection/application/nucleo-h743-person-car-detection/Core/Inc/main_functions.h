/*
 * 	main_functions.h
 *
 * 	Author: Enrique Phan
 * 	Github: PHANzgz
 * 	Date: 22/9/020
 *
 * 	Notes: Ported example from tensorflow repo
 *
 */

#ifndef MAIN_FUNCTIONS_H
#define MAIN_FUNCTIONS_H

#include "main.h"
// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Intializes data (peripherals are set up by CubeMX)

void setup(TIM_HandleTypeDef *htim, UART_HandleTypeDef *huart, DCMI_HandleTypeDef *hdcmi, DMA_HandleTypeDef *hdma_dcmi,
			I2C_HandleTypeDef *hi2c, SPI_HandleTypeDef *hspi);

// Main loop of the program
void loop(TIM_HandleTypeDef *htim, UART_HandleTypeDef *huart, DCMI_HandleTypeDef *hdcmi, DMA_HandleTypeDef *hdma_dcmi,
		I2C_HandleTypeDef *hi2c, SPI_HandleTypeDef *hspi);


#ifdef __cplusplus
}
#endif

#endif // MAIN_FUNCTIONS_H
