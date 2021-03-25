/*
 * Nano 33 BLE Sense
 *      Arudiono nRF52840 module
 *      Common functions
 *
 * Copyright (c) 2020,'21 Kenji Arai / JH1PJL
 *      http://www7b.biglobe.ne.jp/~kenjia/
 *      https://os.mbed.com/users/kenjiArai/
 *      Started:    January   22nd, 2020
 *      Revised:    February  28th, 2021
 *
 */

//  Pre-selection --------------------------------------------------------------
#include    "select_example.h"
//#define USE_COMMON_FUNCTION
#ifdef USE_COMMON_FUNCTION

//  Include --------------------------------------------------------------------
#include    "mbed.h"
#include    "nano33blesense_iodef.h"
#include    "LPS22HB.h"
#include    "LSM9DS1.h"
#include    "HTS221.h"
//#include    "glibr.h" // define as follows

//  Definition -----------------------------------------------------------------
/* APDS-9960 I2C address */
#define APDS9960_I2C_ADDR       0x39
/* APDS-9960 register */
#define APDS9960_ID             0x92
/* Acceptable device IDs */
#define APDS9960_ID_1           0xAB
#define APDS9960_ID_2           0x9C 

//  Constructor ----------------------------------------------------------------

//  RAM ------------------------------------------------------------------------

//  ROM / Constant data --------------------------------------------------------

//  Function prototypes --------------------------------------------------------

//------------------------------------------------------------------------------
//  Control Program
//------------------------------------------------------------------------------
void check_i2c_connected_devices(void)
{
    char dt[2];
    int status;

    // Check I2C line
    i2c.frequency(400000);
    dt[0] = 0;
    print_usb("check I2C device --> START\r\n");
    for (uint8_t i = 0; i < 0x80; i++) {
        int addr = i << 1;
        status = i2c.write(addr, dt, 1, true);
        if (status == 0) {
            print_usb("Get ACK form address = 0x%x\r\n", i);
        }
    }
}

void check_i2c_sensors(void)
{
    char dt[2];

    // LSM9DS1
    dt[0] = WHO_AM_I_XG;
    int addr = LSM9DS1_AG_I2C_ADDR(1);
    i2c.write(addr, dt, 1, true);
    dt[0] = 0;
    i2c.read(addr, dt, 1, false);
    print_usb("LSM9DS1_AG is ");
    if (dt[0] != WHO_AM_I_AG_RSP) {
        print_usb("NOT ");
    }
    print_usb("available -> 0x%x = 0x%x\r\n", addr >> 1, dt[0]);
    dt[0] = WHO_AM_I_M;
    addr = LSM9DS1_M_I2C_ADDR(1);
    i2c.write(addr, dt, 1, true);
    dt[0] = 0;
    i2c.read(addr, dt, 1, false);
    print_usb("LSM9DS1_M is ");
    if (dt[0] != WHO_AM_I_M_RSP) {
        print_usb("NOT ");
    } 
    print_usb("available -> 0x%x = 0x%x\r\n", addr >> 1, dt[0]);
    // LPS22HB
    dt[0] = LPS22HB_WHO_AM_I;
    addr = LPS22HB_G_CHIP_ADDR;
    i2c.write(addr, dt, 1, true);
    dt[0] = 0;
    i2c.read(addr, dt, 1, false);
    print_usb("LPS22HB is ");
    if (dt[0] != I_AM_LPS22HB) {
        print_usb("NOT ");
    } 
    print_usb("available -> 0x%x = 0x%x\r\n", addr >> 1, dt[0]);
    // HTS221
    dt[0] =  HTS221::HTS221_WHO_AM_I;
    addr =  HTS221::HTS221_ADDRESS;
    i2c.write(addr, dt, 1, true);
    dt[0] = 0;
    i2c.read(addr, dt, 1, false);
    print_usb("HTS221 is ");
    if (dt[0] !=  HTS221::WHO_AM_I_VALUE) {
        print_usb("NOT ");
    } 
    print_usb("available -> 0x%x = 0x%x\r\n", addr >> 1, dt[0]);
    // APDS_9960
    dt[0] = APDS9960_ID;
    addr =  APDS9960_I2C_ADDR << 1;
    i2c.write(addr, dt, 1, true);
    dt[0] = 0;
    i2c.read(addr, dt, 1, false);
    print_usb("APDS_9960 is ");
    if (dt[0] == APDS9960_ID_1 || dt[0] == APDS9960_ID_2) {
        ;
    } else {
        print_usb("NOT ");
    } 
    print_usb("available -> 0x%x = 0x%x\r\n", addr >> 1, dt[0]);
}

#endif  // USE_COMMON_FUNCTION