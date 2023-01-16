#include <iostream>
#include "camera_calib.hpp"

int main(int argc, char **argv)
{
    std::string pic_path = "../RGB_camera_calib_img/";
    CamIntrCalib cam_intr_calib(pic_path, 11, 8, 0.02);
    cam_intr_calib.Calibrate();
    cam_intr_calib.CalibrateWithCv();
    return 0;
}