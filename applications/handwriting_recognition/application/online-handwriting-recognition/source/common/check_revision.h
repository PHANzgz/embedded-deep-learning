/*
 * Check Mbed revision
 *
 * Copyright (c) 2019,'20,'21 Kenji Arai / JH1PJL
 *  http://www7b.biglobe.ne.jp/~kenjia/
 *  https://os.mbed.com/users/kenjiArai/
 *      Created:    July      17th, 2019
 *      Revised:    February  28th, 2021
 */

//    RUN ONLY ON mbed-os-6.8.0
//      https://github.com/ARMmbed/mbed-os/releases/tag/mbed-os-6.8.0
#if (MBED_MAJOR_VERSION == 6) &&\
    (MBED_MINOR_VERSION == 8) &&\
    (MBED_PATCH_VERSION == 0)
#else
#   error "Please use mbed-os-6.8.0"
#endif

void print_revision(void)
{
    print_usb("MBED_MAJOR_VERSION = %d, ", MBED_MAJOR_VERSION);
    print_usb("MINOR = %d, ", MBED_MINOR_VERSION);
    print_usb("PATCH = %d\r\n", MBED_PATCH_VERSION);
}