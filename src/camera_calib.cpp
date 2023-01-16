#include "camera_calib.hpp"
// #define DEBUG_CODE
#define USE_CV
bool CamIntrCalib::Calibrate()
{
    if (ReadPics() && GetKeyPoints())
    {
        CalcH();
        // CalcHWithCV();
        ValidateH();

        CalcK();
        CalcT();
        CalcDistCoeff();
        Optimize();

        return true;
    }
    return false;
}
bool CamIntrCalib::ReadPics()
{
    for (int i = 0; i <= 40; ++i)
    {
        std::string single_picture_path = pic_path_ + "/" + std::to_string(i + 100000) + ".png";
        cv::Mat pic = cv::imread(single_picture_path, cv::IMREAD_GRAYSCALE);
        if (pic.empty())
        {
            std::cerr << "read picture failed: " << single_picture_path;
            return false;
        }
#ifdef DEBUG_CODE
        cv::imshow("origin image", pic);
        cv::waitKey(100);
#endif
        calib_pics_.push_back(pic);
        ori_pics_.push_back(pic);
    }

    return true;
}
bool CamIntrCalib::GetKeyPoints()
{
    // world points
    for (size_t i = 0; i < calib_pics_.size(); ++i)
    {
        std::vector<cv::Point2f> points_3ds;
        for (int row = 0; row < points_per_row_; ++row)
        {
            for (int col = 0; col < points_per_col_; ++col)
            {
                cv::Point2f corner_p;
                corner_p.x = row * square_size_;
                corner_p.y = col * square_size_;
                points_3ds.push_back(corner_p);
            }
        }
        points_3d_vec_.emplace_back(std::move(points_3ds));
    }

    // detect chessboard corner
    // https://blog.csdn.net/guduruyu/article/details/69573824
    for (const auto &calib_pic : calib_pics_)
    {
        std::vector<cv::Point2f> corner_pts;
        bool found_flag = cv::findChessboardCorners(
            calib_pic, cv::Size(points_per_col_, points_per_row_), corner_pts,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE); //!!! cv::Size(col,row)
        if (!found_flag)
        {
            std::cerr << "found chessborad corner failed";
            return false;
        }
        cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(calib_pic, corner_pts, cv::Size(11, 11), cv::Size(-1, -1),
                         criteria);
#ifdef DEBUG_CODE
        // 角点绘制
        cv::drawChessboardCorners(calib_pic, cv::Size(points_per_row_, points_per_col_), corner_pts, found_flag);
        cv::imshow("chessboard corner image", calib_pic);
        cv::waitKey(300);
#endif
        points_2d_vec_.push_back(std::move(corner_pts));
    }

    return true;
}
void CamIntrCalib::Normalize(const std::vector<cv::Point2f> &point_vec,
                             std::vector<cv::Point2f> *normed_point_vec,
                             cv::Mat *norm_T)
{
    *norm_T = cv::Mat::eye(3, 3, CV_64F);
    double mean_x = 0;
    double mean_y = 0;
    for (const auto &p : point_vec)
    {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= point_vec.size();
    mean_y /= point_vec.size();
    double mean_dev_x = 0;
    double mean_dev_y = 0;
    for (const auto &p : point_vec)
    {
        mean_dev_x += fabs(p.x - mean_x);
        mean_dev_y += fabs(p.y - mean_y);
    }
    mean_dev_x /= point_vec.size();
    mean_dev_y /= point_vec.size();
    double sx = 1.0 / mean_dev_x;
    double sy = 1.0 / mean_dev_y;
    normed_point_vec->clear();
    for (const auto &p : point_vec)
    {
        cv::Point2f p_tmp;
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normed_point_vec->push_back(p_tmp);
    }
    norm_T->at<double>(0, 0) = sx;
    norm_T->at<double>(0, 2) = -mean_x * sx;
    norm_T->at<double>(1, 1) = sy;
    norm_T->at<double>(1, 2) = -mean_y * sy;
}

void CamIntrCalib::CalcH()
{
    for (size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        const auto &points_2d = points_2d_vec_[id];
        const auto &points_3d = points_3d_vec_[id];
        std::vector<cv::Point2f> normed_points_2d;
        std::vector<cv::Point2f> normed_points_3d;
        cv::Mat norm_T_2d;
        cv::Mat norm_T_3d;
        Normalize(points_2d, &normed_points_2d, &norm_T_2d);
        Normalize(points_3d, &normed_points_3d, &norm_T_3d);

        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);

        int corner_size = normed_points_2d.size();
        if (corner_size < 4)
        {
            std::cerr << "corner size < 4";
            exit(-1);
        }
        cv::Mat A(corner_size * 2, 9, CV_64F, cv::Scalar(0));
        for (int i = 0; i < corner_size; ++i)
        {
            const auto &p_3d = normed_points_3d[i];
            const auto &p_2d = normed_points_2d[i];
            A.at<double>(2 * i, 0) = p_3d.x;
            A.at<double>(2 * i, 1) = p_3d.y;
            A.at<double>(2 * i, 2) = 1;
            A.at<double>(2 * i, 3) = 0;
            A.at<double>(2 * i, 4) = 0;
            A.at<double>(2 * i, 5) = 0;
            A.at<double>(2 * i, 6) = -p_2d.x * p_3d.x;
            A.at<double>(2 * i, 7) = -p_2d.x * p_3d.y;
            A.at<double>(2 * i, 8) = -p_2d.x;

            A.at<double>(2 * i + 1, 0) = 0;
            A.at<double>(2 * i + 1, 1) = 0;
            A.at<double>(2 * i + 1, 2) = 0;
            A.at<double>(2 * i + 1, 3) = p_3d.x;
            A.at<double>(2 * i + 1, 4) = p_3d.y;
            A.at<double>(2 * i + 1, 5) = 1;
            A.at<double>(2 * i + 1, 6) = -p_2d.y * p_3d.x;
            A.at<double>(2 * i + 1, 7) = -p_2d.y * p_3d.y;
            A.at<double>(2 * i + 1, 8) = -p_2d.y;
        }
        cv::Mat U, W, VT;                                                    // A =UWV^T
        cv::SVD::compute(A, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量

        H = VT.row(8).reshape(0, 3);
        cv::Mat norm_T_2d_inv;
        cv::invert(norm_T_2d, norm_T_2d_inv);
        H = norm_T_2d_inv * H * norm_T_3d;
        H_vec_.push_back(H);
        std::cout << "H:\n"
                  << H << std::endl;
        std::cout << "L2 norm: " << cv::norm(H) << std::endl;
    }
}

void CamIntrCalib::CalcHWithCV()
{
    for (size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        const auto &points_2d = points_2d_vec_[id];
        const auto &points_3d = points_3d_vec_[id];

        cv::Mat H = cv::findHomography(points_3d, points_2d);
        H_vec_.push_back(H);
        std::cout << "H:\n"
                  << H << std::endl;
        std::cout << "L2 norm: " << cv::norm(H) << std::endl;
    }
}

void CamIntrCalib::ValidateH()
{
    double reproject_err = 0;
    size_t img_num = points_3d_vec_.size();
    int pts_num = 0;
    std::vector<std::vector<cv::Point2f>> corner_pts_vec;
    for (size_t i = 0; i < img_num; ++i)
    {
        std::vector<cv::Point2f> corner_pts;
        const auto &H = H_vec_[i];

        for (size_t j = 0; j < points_3d_vec_[i].size(); ++j)
        {
            const auto &p_3d = points_3d_vec_[i][j];
            const auto &p_2d = points_2d_vec_[i][j];
            cv::Point2f rep_p;
            double s = H.at<double>(2, 0) * p_3d.x + H.at<double>(2, 1) * p_3d.y + H.at<double>(2, 2);
            rep_p.x = (H.at<double>(0, 0) * p_3d.x + H.at<double>(0, 1) * p_3d.y + H.at<double>(0, 2)) / s;
            rep_p.y = (H.at<double>(1, 0) * p_3d.x + H.at<double>(1, 1) * p_3d.y + H.at<double>(1, 2)) / s;
            std::cout << "(u,v): (" << p_2d.x << ", " << p_2d.y << "); (u',v'): (" << rep_p.x << ", " << rep_p.y << ")" << std::endl;
            reproject_err += (fabs(p_2d.x - rep_p.x) + fabs(p_2d.y - rep_p.y));
            pts_num++;
            corner_pts.emplace_back(rep_p);
        }
        corner_pts_vec.push_back(corner_pts);
    }
    reproject_err /= static_cast<double>(2 * pts_num);
    std::cout << "H reproject error: " << reproject_err << std::endl;
#ifdef DEBUG_CODE

    for (size_t i = 0; i < ori_pics_.size(); ++i)
    {
        cv::drawChessboardCorners(ori_pics_[i], cv::Size(points_per_row_, points_per_col_), corner_pts_vec[i], true);
        cv::imshow("Validate H", ori_pics_[i]);
        cv::waitKey(800);
    }
#endif
}

void CamIntrCalib::CalcK()
{
    cv::Mat A(points_2d_vec_.size() * 2, 6, CV_64F, cv::Scalar(0));
    for (size_t i = 0; i < points_2d_vec_.size(); ++i)
    {
        cv::Mat H = H_vec_[i];
        // 第1列
        double h11 = H.at<double>(0, 0);
        double h21 = H.at<double>(1, 0);
        double h31 = H.at<double>(2, 0);
        // 第2列
        double h12 = H.at<double>(0, 1);
        double h22 = H.at<double>(1, 1);
        double h32 = H.at<double>(2, 1);

        cv::Mat v11 = (cv::Mat_<double>(1, 6) << h11 * h11, h11 * h21 + h11 * h21, h21 * h21, h11 * h31 + h31 * h11, h21 * h31 + h31 * h21 + h31 * h31, h31 * h31);
        cv::Mat v12 = (cv::Mat_<double>(1, 6) << h11 * h12, h11 * h22 + h21 * h12, h21 * h22, h11 * h32 + h31 * h12, h21 * h32 + h31 * h22 + h31 * h32, h31 * h32);
        cv::Mat v22 = (cv::Mat_<double>(1, 6) << h12 * h12, h12 * h22 + h12 * h22, h22 * h22, h12 * h32 + h32 * h12, h22 * h32 + h32 * h22 + h32 * h32, h32 * h32);
        // std::cout << "v11: " << v11 << std::endl;
        v12.copyTo(A.row(2 * i));
        cv::Mat v_tmp = (v11 - v22);
        v_tmp.copyTo(A.row(2 * i + 1));
    }
    // std::cout << "A:\n"
    //           << A << std::endl;
    cv::Mat U, W, VT;              // A =UWV^T
    cv::SVD::compute(A, W, U, VT); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
    std::cout << "VT:\n"
              << VT << std::endl;
    cv::Mat B = VT.row(5);
    std::cout << "B:\n"
              << B << std::endl;
    double B11 = B.at<double>(0, 0);
    double B12 = B.at<double>(0, 1);
    double B22 = B.at<double>(0, 2);
    double B13 = B.at<double>(0, 3);
    double B23 = B.at<double>(0, 4);
    double B33 = B.at<double>(0, 5);
    std::cout << "B: " << B11 << "," << B12 << "," << B22 << "," << B13 << "," << B23 << "," << B33 << std::endl;

    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = sqrt(lambda / B11);
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    std::cout << "K coeff: " << alpha << " , " << beta << " , " << gamma << " , " << lambda << " , " << u0 << " , " << v0 << std::endl;

    gamma = 0;

    K_.at<double>(0, 0) = alpha;
    K_.at<double>(0, 1) = gamma;
    K_.at<double>(0, 2) = u0;
    K_.at<double>(1, 1) = beta;
    K_.at<double>(1, 2) = v0;
    std::cout << "K:\n"
              << K_ << std::endl;
}
void CamIntrCalib::CalcT()
{

    cv::Mat K_inverse;
    cv::invert(K_, K_inverse);
    for (const auto &H : H_vec_)
    {
        cv::Mat R_t = K_inverse * H;
        cv::Vec3d r1(R_t.at<double>(0, 0), R_t.at<double>(1, 0), R_t.at<double>(2, 0));
        cv::Vec3d r2(R_t.at<double>(0, 1), R_t.at<double>(1, 1), R_t.at<double>(2, 1));
        cv::Vec3d r3 = r1.cross(r2);
        cv::Mat Q = cv::Mat::eye(3, 3, CV_64F);
        Q.at<double>(0, 0) = r1(0);
        Q.at<double>(1, 0) = r1(1);
        Q.at<double>(2, 0) = r1(2);
        Q.at<double>(0, 1) = r2(0);
        Q.at<double>(1, 1) = r2(1);
        Q.at<double>(2, 1) = r2(2);
        Q.at<double>(0, 2) = r3(0);
        Q.at<double>(1, 2) = r3(1);
        Q.at<double>(2, 2) = r3(2);
        cv::Mat norm_Q;
        cv::normalize(Q, norm_Q);
        cv::Mat U, W, VT;                                                         // A =UWV^T
        cv::SVD::compute(norm_Q, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        cv::Mat R = U * VT;
        R_vec_.push_back(R);
        cv::Mat R_T;
        cv::transpose(R, R_T);
        // std::cout << "R*RT:\n"
        //           << R * R_T << std::endl;
        cv::Mat t = cv::Mat::eye(3, 1, CV_64F);
        R_t.col(2).copyTo(t.col(0));
        t_vec_.push_back(t);
    }
}
// use one pic to calc
void CamIntrCalib::CalcDistCoeff()
{
    // calc ideal corner point Puv = KTPw
    std::vector<double> r2_vec;
    std::vector<cv::Point2f> ideal_point_vec;

    cv::Mat R = R_vec_[0];
    cv::Mat t = t_vec_[0];
    std::vector<cv::Point2f> point_3ds = points_3d_vec_[0];
    for (const auto &p : point_3ds)
    {
        cv::Mat p_3d = (cv::Mat_<double>(3, 1) << p.x, p.y, 0);
        cv::Mat p_pic = R * p_3d + t;
        p_pic.at<double>(0, 0) = p_pic.at<double>(0, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(1, 0) = p_pic.at<double>(1, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(2, 0) = 1;
        double x = p_pic.at<double>(0, 0);
        double y = p_pic.at<double>(1, 0);
        double r2 = x * x + y * y;
        r2_vec.push_back(r2);

        cv::Mat p_uv = K_ * p_pic;
        ideal_point_vec.emplace_back(p_uv.at<double>(0, 0), p_uv.at<double>(1, 0));
    }

    // points_2d_vec_ is distort uv
    std::vector<cv::Point2f> dist_point_vec = points_2d_vec_[0];
    double u0 = K_.at<double>(0, 2);
    double v0 = K_.at<double>(1, 2);
    cv::Mat D = cv::Mat::eye(ideal_point_vec.size() * 2, 2, CV_64F);
    cv::Mat d = cv::Mat::eye(ideal_point_vec.size() * 2, 1, CV_64F);
    for (size_t i = 0; i < ideal_point_vec.size(); ++i)
    {
        double r2 = r2_vec[i];
        cv::Point2f distort_p = dist_point_vec[i];
        cv::Point2f ideal_p = ideal_point_vec[i];
        D.at<double>(2 * i, 0) = (ideal_p.x - u0) * r2;
        D.at<double>(2 * i, 1) = (ideal_p.x - u0) * r2 * r2;
        D.at<double>(2 * i + 1, 0) = (ideal_p.y - v0) * r2;
        D.at<double>(2 * i + 1, 1) = (ideal_p.y - v0) * r2 * r2;
        d.at<double>(0, 0) = distort_p.x - ideal_p.x;
        d.at<double>(1, 0) = distort_p.y - ideal_p.y;
    }
    cv::Mat DT;
    cv::transpose(D, DT);
    cv::Mat DTD_inverse;
    cv::invert(DT * D, DTD_inverse);
    dist_coef_ = DTD_inverse * DT * d;
    std::cout << "distort coeff: " << dist_coef_.at<double>(0, 0) << ", " << dist_coef_.at<double>(1, 0) << std::endl;
}

struct CamIntrCalib::ReprojErr
{
public:
    ReprojErr(const cv::Point2f &observe_p_2d_)
        : observe_p_2d(observe_p_2d_) {}
    template <typename T>
    bool operator()(const T *const camera, const T *const point, const T *const K, const T *const dist_coeff, T *residual) const
    {
        // ！！！ camera传入的并不是6*1的矩阵，只是取前面6个元素
        //  camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T x = p[0] / p[2];
        T y = p[1] / p[2];
        T r2 = x * x + y * y;

        const T &alpha = K[0];
        const T &beta = K[1];
        const T &u0 = K[2];
        const T &v0 = K[3];

        const T &k1 = dist_coeff[0];
        const T &k2 = dist_coeff[1];

        T x_dist = x * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2);
        T y_dist = y * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2);

        const T u_dist = alpha * x_dist + u0;
        const T v_dist = beta * y_dist + v0;

        residual[0] = u_dist - static_cast<T>(observe_p_2d.x);
        residual[1] = v_dist - static_cast<T>(observe_p_2d.y);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2f &observe_p_2d_)
    {
        return new ceres::AutoDiffCostFunction<ReprojErr, 2, 6, 3, 4, 2>(new ReprojErr(observe_p_2d_));
    }

private:
    const cv::Point2f observe_p_2d;
};

void CamIntrCalib::Optimize()
{
    ceres::Problem problem;
    int pic_num = points_3d_vec_.size();
    int every_pic_p_num = points_3d_vec_[0].size();
    double *K_para = new double[4];
    *K_para = K_.at<double>(0, 0);
    *(K_para + 1) = K_.at<double>(1, 1);
    *(K_para + 2) = K_.at<double>(0, 2);
    *(K_para + 3) = K_.at<double>(1, 2);
    double *coeff_para = new double[2];
    *coeff_para = dist_coef_.at<double>(0, 0);
    *(coeff_para + 1) = dist_coef_.at<double>(1, 0);
    double *cam_para = new double[6 * pic_num];
    double *point = new double[3 * pic_num * every_pic_p_num];
    int pic_index = 0;
    int point_index = 0;
    for (int i = 0; i < pic_num; ++i)
    {
        const cv::Mat &R = R_vec_[i];
        const cv::Mat &t = t_vec_[i];
        cv::Mat angle_axis;
        cv::Rodrigues(R, angle_axis);

        *(cam_para + 6 * pic_index) = angle_axis.at<double>(0, 0);
        *(cam_para + 6 * pic_index + 1) = angle_axis.at<double>(1, 0);
        *(cam_para + 6 * pic_index + 2) = angle_axis.at<double>(2, 0);
        *(cam_para + 6 * pic_index + 3) = t.at<double>(0, 0);
        *(cam_para + 6 * pic_index + 4) = t.at<double>(1, 0);
        *(cam_para + 6 * pic_index + 5) = t.at<double>(2, 0);

        const std::vector<cv::Point2f> &points_3ds = points_3d_vec_[i];
        for (size_t j = 0; j < points_3ds.size(); ++j)
        {
            *(point + 3 * point_index) = points_3ds[j].x;
            *(point + 3 * point_index + 1) = points_3ds[j].y;
            *(point + 3 * point_index + 2) = 0;
            ++point_index;
        }
        ++pic_index;
    }

    pic_index = 0;
    point_index = 0;
    for (int i = 0; i < pic_num; ++i)
    {
        const std::vector<cv::Point2f> &points_3ds = points_3d_vec_[i];
        const std::vector<cv::Point2f> &points_2ds = points_2d_vec_[i];
        for (size_t j = 0; j < points_3ds.size(); ++j)
        {
            double *cam_para_now = cam_para + 6 * pic_index;
            double *point_now = point + 3 * point_index;
            ceres::CostFunction *cost_function =
                ReprojErr::Create(points_2ds[j]);
            problem.AddResidualBlock(cost_function, nullptr, cam_para_now, point_now, K_para, coeff_para);
            ++point_index;
        }
        ++pic_index;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "origin K:" << K_.at<double>(0, 0) << " " << K_.at<double>(1, 1) << " " << K_.at<double>(0, 2) << " " << K_.at<double>(1, 2) << std::endl;
    std::cout << "origin dist_coeff:" << dist_coef_.at<double>(0, 0) << " " << dist_coef_.at<double>(1, 0) << std::endl;
    K_.at<double>(0, 0) = *(K_para);
    K_.at<double>(1, 1) = *(K_para + 1);
    K_.at<double>(0, 2) = *(K_para + 2);
    K_.at<double>(1, 2) = *(K_para + 3);
    dist_coef_.at<double>(0, 0) = *(coeff_para);
    dist_coef_.at<double>(1, 0) = *(coeff_para + 1);
    std::cout << "optimize K:" << K_.at<double>(0, 0) << " " << K_.at<double>(1, 1) << " " << K_.at<double>(0, 2) << " " << K_.at<double>(1, 2) << std::endl;
    std::cout << "optimize dist_coeff:" << dist_coef_.at<double>(0, 0) << " " << dist_coef_.at<double>(1, 0) << std::endl;
}