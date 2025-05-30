#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace auto_aim {

class PNPSolver {
public:
    PNPSolver(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs);

    // 解算位姿
    void solvePose(const std::vector<cv::Point2f>& img_points, 
                  cv::Mat& rvec, cv::Mat& tvec,
                  double& yaw_out) const;
    
    // 获取欧拉角 - 使用旋转向量
    void getEulerAngles(const cv::Mat& rvec, double& yaw, double& pitch, double& roll) const;
    
    // 获取欧拉角 - 直接使用旋转矩阵的重载版本
    // 删除了默认参数，必须明确提供isRotMatrix参数
    void getEulerAngles(const cv::Mat& rmat, double& yaw, double& pitch, double& roll, bool isRotMatrix) const;
    
    // 获取缓存的旋转矩阵
    const cv::Mat& getRotationMatrix(const cv::Mat& rvec) const;

    // 在图像上绘制位姿信息
    void drawPoseInfo(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec) const;

private:
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
    std::vector<cv::Point3f> object_points_;
    
    // 缓存旋转矩阵
    mutable cv::Mat last_rvec_;
    mutable cv::Mat cached_rmat_;
    mutable bool rmat_cached_ = false;
};

} // namespace auto_aim