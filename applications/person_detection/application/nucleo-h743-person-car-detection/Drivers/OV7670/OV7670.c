/*
 * ov7670.c
 *
 *  Created on: 2017/08/25
 *      Author: take-iwiw
 *
 *  Modified on: 2020/09/17
 *  Modified by: Enrique Phan
 *  Github: PHANzgz
 *
 */
#include <stdio.h>
#include "main.h"
#include "OV7670.h"
#include <stdint.h>
#include "OV7670_REG.h"


/*** Internal Const Values, Macros ***/


/*** Internal Static Variables ***/
static DCMI_HandleTypeDef *sp_hdcmi;
static DMA_HandleTypeDef  *sp_hdma_dcmi;
static I2C_HandleTypeDef  *sp_hi2c;
static uint32_t    s_destAddressForContiuousMode;
static void (* s_cbHsync)(uint32_t h);
static void (* s_cbVsync)(uint32_t v);
static void (* s_cbFrame)();
static uint32_t s_currentH;
static uint32_t s_currentV;
static uint8_t capture = 0;
static uint8_t imgMode = OV7670_MODE_QVGA_RGB565;

/*** Internal Function Declarations ***/
//static HAL_StatusTypeDef ov7670_write(uint8_t regAddr, uint8_t data);
//static HAL_StatusTypeDef ov7670_read(uint8_t regAddr, uint8_t *data);

/*** External Function Defines ***/
HAL_StatusTypeDef ov7670_init(DCMI_HandleTypeDef *p_hdcmi, DMA_HandleTypeDef *p_hdma_dcmi, I2C_HandleTypeDef *p_hi2c){
  sp_hdcmi     = p_hdcmi;
  sp_hdma_dcmi = p_hdma_dcmi;
  sp_hi2c      = p_hi2c;
  s_destAddressForContiuousMode = 0;

  HAL_GPIO_WritePin(CAMERA_RESET_GPIO_Port, CAMERA_RESET_Pin, GPIO_PIN_RESET);
  HAL_Delay(100);
  HAL_GPIO_WritePin(CAMERA_RESET_GPIO_Port, CAMERA_RESET_Pin, GPIO_PIN_SET);
  HAL_Delay(100);

  ov7670_write(0x12, 0x80);  // RESET
  HAL_Delay(30);

  //uint8_t buffer[4];
  //ov7670_read(0x0b, buffer);

  return HAL_OK;
}

HAL_StatusTypeDef ov7670_config(uint8_t mode){
  ov7670_stopCap();
  ov7670_write(0x12, 0x80);  // RESET
  HAL_Delay(30);
  for(int i = 0; OV7670_reg[i][0] != REG_EOT; i++) {
    ov7670_write(OV7670_reg[i][0], OV7670_reg[i][1]);
    HAL_Delay(1);
  }
  if(mode == OV7670_MODE_QVGA_YUV){
	  ov7670_write(0x12, 0x10); // QVGA, YUV
	  ov7670_write(0x40, 0xc0); // 00-FF, No RGB

	  imgMode = OV7670_MODE_QVGA_YUV;
  }
  return HAL_OK;
}

// Capture modes are OV7670_CAP_SINGLE_FRAME or OV7670_CAP_CONTINUOUS
// Image modes are OV7670_MODE_QVGA_RGB565 or OV7670_MODE_QVGA_YUV
HAL_StatusTypeDef ov7670_startCap(uint32_t capMode, uint32_t destAddress){

	ov7670_stopCap();
	if (capMode == OV7670_CAP_CONTINUOUS) {
		/* note: continuous mode automatically invokes DCMI, but DMA needs to be invoked manually */
		s_destAddressForContiuousMode = destAddress;
		HAL_DCMI_Start_DMA(sp_hdcmi, DCMI_MODE_CONTINUOUS, destAddress, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/imgMode);
	} else if (capMode == OV7670_CAP_SINGLE_FRAME) {
		s_destAddressForContiuousMode = 0;
		capture = 1;
		HAL_DCMI_Start_DMA(sp_hdcmi, DCMI_MODE_SNAPSHOT, destAddress, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/imgMode);
	}

	return HAL_OK;
}

HAL_StatusTypeDef ov7670_stopCap(){
  HAL_DCMI_Stop(sp_hdcmi);
//  HAL_Delay(30);
  return HAL_OK;
}

void ov7670_registerCallback(void (*cbHsync)(), void (*cbVsync)(), void (*cbFrame)()){
  s_cbHsync = cbHsync;
  s_cbVsync = cbVsync;
  s_cbFrame = cbFrame;
}

void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi){
//  printf("FRAME %d\n", HAL_GetTick());
  if(s_cbFrame)s_cbFrame();
  if(s_destAddressForContiuousMode != 0) {
    HAL_DMA_Start_IT(hdcmi->DMA_Handle, (uint32_t)&hdcmi->Instance->DR, s_destAddressForContiuousMode, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
  }
  s_currentV++;
  s_currentH = 0;
}

void HAL_DCMI_VsyncEventCallback(DCMI_HandleTypeDef *hdcmi){
	if(capture) if(s_cbVsync)s_cbVsync(s_currentV);
	capture = 0;
//  printf("VSYNC %d\n", HAL_GetTick());
//  HAL_DMA_Start_IT(hdcmi->DMA_Handle, (uint32_t)&hdcmi->Instance->DR, s_destAddressForContiuousMode, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
}

//void HAL_DCMI_LineEventCallback(DCMI_HandleTypeDef *hdcmi)
//{
////  printf("HSYNC %d\n", HAL_GetTick());
//  if(s_cbHsync)s_cbHsync(s_currentH);
//  s_currentH++;
//}

/*** Internal Function Defines ***/
HAL_StatusTypeDef ov7670_write(uint8_t regAddr, uint8_t data){
  HAL_StatusTypeDef ret;
  do {
    ret = HAL_I2C_Mem_Write(sp_hi2c, SLAVE_ADDR, regAddr, I2C_MEMADD_SIZE_8BIT, &data, 1, 100);
	//ret = HAL_I2C_Master_Transmit(sp_hi2c, SLAVE_ADDR, data, 1, 100);
  } while (ret != HAL_OK && 0);
  return ret;
}

HAL_StatusTypeDef ov7670_read(uint8_t regAddr, uint8_t *data){
  HAL_StatusTypeDef ret;
  do {
    // HAL_I2C_Mem_Read doesn't work (because of SCCB protocol(doesn't have ack))? */
	//    ret = HAL_I2C_Mem_Read(sp_hi2c, SLAVE_ADDR, regAddr, I2C_MEMADD_SIZE_8BIT, data, 1, 1000);
    ret = HAL_I2C_Master_Transmit(sp_hi2c, SLAVE_ADDR+1, &regAddr, 1, 100);
    ret |= HAL_I2C_Master_Receive(sp_hi2c, SLAVE_ADDR+1, data, 1, 100);
  } while (ret != HAL_OK && 0);
  return ret;
}
