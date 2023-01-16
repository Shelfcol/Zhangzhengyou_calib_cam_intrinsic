#pragma once
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
class CamIntrCalib
{
public:
    explicit CamIntrCalib(const std::string &pic_path, const int points_per_row,
                          const int points_per_col,
                          const double square_size);
    bool Calibrate();
    bool CalibrateWithCv();
    bool ReadPics();
    bool GetKeyPoints();
    void CalcH();
    void CalcHWithCV();
    void ValidateH();
    void CalcK();
    void CalcT();
    void CalcDistCoeff();
    void Optimize();
    void Transform2dTo3dPts(const std::vector<std::vector<cv::Point2f>> &points_3d_vec,
                            std::vector<std::vector<cv::Point3f>> *points_3ds);
    void Normalize(const std::vector<cv::Point2f> &point_vec,
                   std::vector<cv::Point2f> *normed_point_vec,
                   cv::Mat *norm_T);
    cv::Point2f ReprojectPoint(const cv::Point3f &p_3d, const cv::Mat &R, const cv::Mat &t,
                               const cv::Mat &K, const double k1, const double k2);
    double CalcRepjErr();
    double CalcDiff(const std::vector<cv::Point2f> &points_2d_vec1,
                    const std::vector<cv::Point2f> &points_2d_vec2);
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