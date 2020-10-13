/*
 * ov7670.h
 *
 *  Created on: 2017/08/25
 *      Author: take-iwiw
 *
 *  Modified on: 2020/09/17
 *  Modified by: Enrique Phan
 *  Github: PHANzgz
 *
 */

#ifndef OV7670_OV7670_H_
#define OV7670_OV7670_H_

#ifdef __cplusplus
extern "C" {
#endif

// The following two values are for convenience in startCap function.
#define OV7670_MODE_QVGA_RGB565 2
#define OV7670_MODE_QVGA_YUV    4

#define OV7670_CAP_CONTINUOUS   0
#define OV7670_CAP_SINGLE_FRAME 1

#define SLAVE_ADDR 0x42
#define OV7670_QVGA_WIDTH  320
#define OV7670_QVGA_HEIGHT 240

HAL_StatusTypeDef ov7670_init(DCMI_HandleTypeDef *p_hdcmi, DMA_HandleTypeDef *p_hdma_dcmi, I2C_HandleTypeDef *p_hi2c);
HAL_StatusTypeDef ov7670_config(uint8_t mode);
HAL_StatusTypeDef ov7670_startCap(uint32_t capMode, uint32_t destAddress);
HAL_StatusTypeDef ov7670_stopCap();
void ov7670_registerCallback(void (*cbHsync)(), void (*cbVsync)(), void (* cbFrame)());

HAL_StatusTypeDef ov7670_write(uint8_t regAddr, uint8_t data);
HAL_StatusTypeDef ov7670_read(uint8_t regAddr, uint8_t *data);

#ifdef __cplusplus
}
#endif

#endif /* OV7670_OV7670_H_ */

