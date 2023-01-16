#pragma once
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
class CamIntrCalib
{
public:
    explicit CamIntrCalib(const std::string &pic_path, const int points_per_row,
                          const int points_per_col,
                          const double square_size) : pic_path_(pic_path),
                                                      points_per_row_(points_per_row),
                                                      points_per_col_(points_per_col),
                                                      square_size_(square_size)
    {
        K_ = cv::Mat::eye(3, 3, CV_64F);
        dist_coef_ = cv::Mat::zeros(4, 1, CV_64F);
    }
    bool Calibrate();
    bool ReadPics();
    bool GetKeyPoints();
    void CalcH();
    void CalcHWithCV();
    void ValidateH();
    void CalcK();
    void CalcT();
    void CalcDistCoeff();
    void Optimize();
    void Normalize(const std::vector<cv::Point2f> &point_vec,
                   std::vector<cv::Point2f> *normed_point_vec,
                   cv::Mat *norm_T);
    struct ReprojErr;

private:
    const std::string pic_path_;
    const int points_per_row_;
    const int points_per_col_;
    const double square_size_;

    std::vector<cv::Mat> calib_pics_;
    std::vector<cv::Mat> ori_pics_;

    cv::Mat K_;
    cv::Mat H_;
    cv::Mat dist_coef_;
    std::vector<cv::Mat> H_vec_;
    std::vector<cv::Mat> R_vec_;
    std::vector<cv::Mat> t_vec_;

    std::vector<std::vector<cv::Point2f>> points_3d_vec_; // z=0
    std::vector<std::vector<cv::Point2f>> points_2d_vec_;
};