/*
 * 	main_functions.cpp
 *
 * 	Author: Enrique Phan
 * 	Github: PHANzgz
 * 	Date: 22/9/020
 *
 * 	Notes:
 *
 */

/* Private includes ----------------------------------------------------------*/
#include "string.h"

#include "main.h"
#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
//#include "test_image_565.h"

#include "../../Drivers/OV7670/OV7670.h"
#include "../../Drivers/ILI9341_Serial/ILI9341.h"
#include "../../Drivers/ILI9341_Serial/ILI9341_GFX.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


/* Private typedef -----------------------------------------------------------*/


/* Private define ------------------------------------------------------------*/


/* Private macro -------------------------------------------------------------*/

/* Private function definition -----------------------------------------------*/
static void onFrameCallback();
void Draw_Image(const unsigned char* img);


/* Private variables ---------------------------------------------------------*/
namespace {
// Tensorflow variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 364*1024; // Experimental minimum
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Other useful variables
uint32_t frame_buffer[OV7670_QVGA_HEIGHT*OV7670_QVGA_WIDTH/(2*2)]; // Horizontal resolution is halved to reduce RAM usage so we divide by 2
uint8_t new_capture = 0;

}  //namespace

/* Private user code ---------------------------------------------------------*/

void setup(TIM_HandleTypeDef *htim, UART_HandleTypeDef *huart, DCMI_HandleTypeDef *hdcmi, DMA_HandleTypeDef *hdma_dcmi,
		I2C_HandleTypeDef *hi2c, SPI_HandleTypeDef *hspi){

	HAL_GPIO_WritePin(CAMERA_PWR_DWN_GPIO_Port, CAMERA_PWR_DWN_Pin, GPIO_PIN_RESET); // Turn on camera
	ov7670_init(hdcmi, hdma_dcmi, hi2c);
	ov7670_config(OV7670_MODE_QVGA_RGB565);
	ov7670_registerCallback(NULL, &onFrameCallback, NULL);
	ILI9341_Init(hspi);
	HAL_TIM_Base_Start(htim); // Init timer for monitoring

	// TENSORFLOW
	// Set up logging.
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. Lightweight operation.
	model = tflite::GetModel(uc_final_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
				"Model provided is schema version %d not equal "
				"to supported version %d.",
				model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	// Pull in only the operation implementations we need
	static tflite::MicroMutableOpResolver<8> micro_op_resolver(error_reporter);
	if (micro_op_resolver.AddResizeNearestNeighbor() != kTfLiteOk) return;
	if (micro_op_resolver.AddMul() != kTfLiteOk) return;
	if (micro_op_resolver.AddSub() != kTfLiteOk) return;
	if (micro_op_resolver.AddPad() != kTfLiteOk) return;
	if (micro_op_resolver.AddConv2D() != kTfLiteOk) return;
	if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) return;
	if (micro_op_resolver.AddAveragePool2D() != kTfLiteOk) return;
	if (micro_op_resolver.AddLogistic() != kTfLiteOk) return;

	static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver,
			tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}

	// Obtain pointers to the model's input and output tensors.
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

	// Get number of elements in input tensor
	//uint32_t num_elements = model_input->bytes / sizeof(uint8_t);

	//Capture first image
	ov7670_startCap(OV7670_CAP_SINGLE_FRAME, (uint32_t)frame_buffer);

}

void loop(TIM_HandleTypeDef *htim, UART_HandleTypeDef *huart, DCMI_HandleTypeDef *hdcmi, DMA_HandleTypeDef *hdma_dcmi,
		I2C_HandleTypeDef *hi2c, SPI_HandleTypeDef *hspi){


	static uint32_t timestamp = 0;
	static char buf[100];
	static uint8_t buf_len;
	static int8_t car_score = -128;
	static int8_t neg_score = -128;
	static int8_t person_score = -128;

	if (new_capture){
		new_capture=0;

		// TENSORFLOW
		// Fill input buffer
		uint16_t *pixel_pointer = (uint16_t *)frame_buffer;
		uint32_t input_ix = 0;

		for (uint32_t pix=0; pix<OV7670_QVGA_HEIGHT*OV7670_QVGA_WIDTH/2; pix++){
			// Convert from RGB55 to RGB888 and int8 range
			uint16_t color = pixel_pointer[pix];
			int16_t r = ((color & 0xF800) >> 11)*255/0x1F - 128;
			int16_t g = ((color & 0x07E0) >> 5)*255/0x3F - 128;
			int16_t b = ((color & 0x001F) >> 0)*255/0x1F - 128;

			model_input->data.int8[input_ix] =   (int8_t) r;
			model_input->data.int8[input_ix+1] = (int8_t) g;
			model_input->data.int8[input_ix+2] = (int8_t) b;

			input_ix += 3;
		}

		// Run inference, measure time and report any error
		timestamp = htim->Instance->CNT;
		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
			return;
		}
		timestamp = htim->Instance->CNT - timestamp;
		car_score = model_output->data.int8[0];
		neg_score = model_output->data.int8[1];
		person_score = model_output->data.int8[2];
		// END TENSORFLOW

		// Display inference information
		Draw_Image((unsigned char*) frame_buffer);
		ILI9341_Set_Rotation(SCREEN_HORIZONTAL_2);
		if (person_score > 0) ILI9341_Draw_Text("PERSON", 180, 210, GREEN, 2, BLACK);
		else ILI9341_Draw_Text("PERSON", 180, 210, RED, 2, BLACK);
		if (car_score >  0) ILI9341_Draw_Text("CAR", 80, 210, GREEN, 2, BLACK);
		else ILI9341_Draw_Text("CAR", 80, 210, RED, 2, BLACK);

		// Print inference info
		buf_len = sprintf(buf,
				"car: %+*d, neg: %+*d, person: %+*d | Duration: %lu ms\r\n",
				4,car_score, 4,neg_score, 4,person_score , timestamp/1000);
		HAL_UART_Transmit(huart, (uint8_t *)buf, buf_len, 100);

		// Capture a new image
		ov7670_startCap(OV7670_CAP_SINGLE_FRAME, (uint32_t)frame_buffer);

	}
}

static void onFrameCallback(){
	new_capture = 1;

}

void Draw_Image(const unsigned char* img){
	ILI9341_Set_Rotation(SCREEN_HORIZONTAL_1);
	ILI9341_Set_Address(0, 0, ILI9341_SCREEN_WIDTH, ILI9341_SCREEN_HEIGHT);

	HAL_GPIO_WritePin(LCD_DC_PORT, LCD_DC_PIN, GPIO_PIN_SET);
	HAL_GPIO_WritePin(LCD_CS_PORT, LCD_CS_PIN, GPIO_PIN_RESET);

	const uint16_t buf_size = BURST_MAX_SIZE;
	unsigned char tx_buf[buf_size];
	uint32_t counter = 0;
	for(uint32_t i = 0; i < (uint32_t)ILI9341_SCREEN_HEIGHT*ILI9341_SCREEN_WIDTH*2/buf_size; i++)
	{
		for(uint32_t k = 0; k< buf_size; k+=4)
		{
			tx_buf[k]	= img[counter+k/2+1];	// Little endian
			tx_buf[k+1]	= img[counter+k/2];
			tx_buf[k+2]	= img[counter+k/2+1];	// Horizontal resolution is halved so we copy same pixel twice
			tx_buf[k+3]	= img[counter+k/2];
		}
		HAL_SPI_Transmit(sp_hspi, (unsigned char*)tx_buf, buf_size, 10);
		counter += buf_size/2;
	}
	HAL_GPIO_WritePin(LCD_CS_PORT, LCD_CS_PIN, GPIO_PIN_SET);
}









