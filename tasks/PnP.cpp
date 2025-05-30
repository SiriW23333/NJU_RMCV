#include "PnP.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "img_tools.hpp"

namespace auto_aim {

// 装甲板参数
static const double LIGHTBAR_LENGTH = 0.056; // 灯条长度    单位：米
static const double ARMOR_WIDTH = 0.135;     // 装甲板宽度  单位：米

PNPSolver::PNPSolver(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs) 
    : camera_matrix_(camera_matrix), distort_coeffs_(distort_coeffs) {
    // 初始化装甲板坐标系下4个点的坐标
    object_points_ = {
        { -ARMOR_WIDTH/2, -LIGHTBAR_LENGTH/2, 0 },  // 点 1
        { ARMOR_WIDTH/2, -LIGHTBAR_LENGTH/2, 0 },  // 点 2
        { ARMOR_WIDTH/2, LIGHTBAR_LENGTH/2, 0 },  // 点 3
        { -ARMOR_WIDTH/2, LIGHTBAR_LENGTH/2, 0 }   // 点 4
    };
}

void PNPSolver::solvePose(const std::vector<cv::Point2f>& img_points, 
                         cv::Mat& rvec, cv::Mat& tvec,
                         double& yaw_out) const {
    // 确保rvec和tvec已正确分配
    if (rvec.empty())
        rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    if (tvec.empty())
        tvec = cv::Mat::zeros(3, 1, CV_64FC1);
        
    cv::solvePnP(object_points_, img_points, camera_matrix_, distort_coeffs_, 
                rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    // 计算yaw角
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    yaw_out = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
}

void PNPSolver::getEulerAngles(const cv::Mat& rvec, double& yaw, double& pitch, double& roll) const {
    // 转换旋转向量为旋转矩阵
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    
    // 计算欧拉角
    yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
    pitch = atan2(-rmat.at<double>(2, 0), 
        sqrt(rmat.at<double>(2, 1) * rmat.at<double>(2, 1) + rmat.at<double>(2, 2) * rmat.at<double>(2, 2)));
    roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
}

} // namespace auto_aim