#include "ekf.hpp"

#include <cmath>

/*
设计思路：
zk:(xa, za, ya, yaw, r)   a作为下标代表实际装甲板位置 yaw为偏航角 r为旋转半径

xk:(xc, vxc, axc, zc, vzc, azc, yc, vyc, ayc, yaw, wyaw, r)           c作为下标代表装甲板旋转中心位置
则 xa = xc - rsin(yaw); za = zc - rcos(yaw); ya = yc; yaw = yaw, r = r
xc+1 = xc + vxcdt + 0.5 axcdt^2; zc+1 = zc + vzccdt + 0.5 azdt^2; yc+1 = yc + vyc*dt + 0.5*ayc*dt*dt; yaw+1 = yaw + wyawdt; r = r
vxc+1 = vxc; vzc+1 = vzc; vyc+1 = vyc; wyaw+1 = wyaw
axc+1 = axc; azc+1 = azc; ayc+1 = ayc
*/

constexpr int STATE_DIM = 12;
constexpr int MEAS_DIM = 5;

namespace auto_aim
{
    EKF::EKF()
    {
        x_ = Eigen::VectorXd::Zero(STATE_DIM);
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-2;
        Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-3;
        Q_.setZero();
        Q_(0, 0) = Q_(3, 3) = Q_(6, 6) = 1e-3;
        Q_(1, 1) = Q_(4, 4) = Q_(7, 7) = 5e-3;
        Q_(5, 5) = 1e-3;
        Q_(2, 2) = 1e-4;
        Q_(6, 6) = 5e-3;
        Q_(7, 7) = 5e-3;
        Q_(8, 8) = 5e-3;
        Q_(9, 9) = 5e-3;
        Q_(10, 10) = 1e-2;
        Q_(11, 11) = 1e-2;
        R_ = Eigen::MatrixXd::Identity(MEAS_DIM, MEAS_DIM) * 5e-2;
        R_(0, 0) = R_(1, 1) = 1e-3; // xa, za 观测噪声降低，增加观测影响权重
        R_(3, 3) = 1e-3;            // yaw 的观测也信任更多
    }
    void EKF::init(double dt)
    {
        dt_ = dt;
    }
    void EKF::setState(const Eigen::VectorXd &x0)
    {
        x_ = x0;
    }
    void EKF::reset(const Eigen::VectorXd &x0)
    {
        x_ = x0;
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-2;
    }
    Eigen::VectorXd EKF::Onlypredict()
    {
        Eigen::VectorXd predict = f(x_);
        return predict;
    }
    Eigen::VectorXd EKF::predict()
    {
        // State prediction
        x_ = f(x_);
        Eigen::MatrixXd F = computeFJacobian(x_);
        P_ = F * P_ * F.transpose() + Q_;
        return x_;
    }

    Eigen::VectorXd EKF::update(const Eigen::VectorXd &z)
    {
        Eigen::VectorXd x_prev = x_; // 保存上一步状态
        Eigen::VectorXd z_pred = h(x_);
        Eigen::VectorXd y = z - z_pred;

        // --- 异常值剔除：如果观测残差过大则丢弃本次观测 ---
        double threshold = 0.05; // 可根据实际情况调整
        if (y.norm() > threshold) {
            // 观测异常，直接返回当前状态，不更新
            return x_;
        }
        // ---------------------------------------------------

        Eigen::MatrixXd H = computeHJacobian(x_);
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H) * P_;

        return x_;
    }

    Eigen::VectorXd EKF::f(const Eigen::VectorXd &x)
    {
        // xk:(xc, vxc, axc, zc, vzc, azc, yc, vyc, ayc, yaw, wyaw, r)
        Eigen::VectorXd x_pred = x;
        x_pred(0) += x(1) * dt_ + 0.5 * x(2) * dt_ * dt_;
        x_pred(1) += x(2) * dt_;
        x_pred(3) += x(4) * dt_ + 0.5 * x(5) * dt_ * dt_;
        x_pred(4) += x(5) * dt_;
        x_pred(6) += x(7) * dt_ + 0.5 * x(8) * dt_ * dt_;
        x_pred(7) += x(8) * dt_;
        x_pred(9) += x(10) * dt_;
        return x_pred;
    }

    Eigen::MatrixXd EKF::computeFJacobian(const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        F(0, 1) = dt_;
        F(0, 2) = 0.5 * dt_ * dt_;
        F(1, 2) = dt_;

        F(3, 4) = dt_;
        F(3, 5) = 0.5 * dt_ * dt_;
        F(4, 5) = dt_;

        F(6, 7) = dt_;
        F(6, 8) = 0.5 * dt_ * dt_;
        F(7, 8) = dt_;

        F(9, 10) = dt_;
        return F;
    }

    Eigen::VectorXd EKF::h(const Eigen::VectorXd &x)
    {
        Eigen::VectorXd z_pred(MEAS_DIM);
        double xc = x(0), zc = x(3), ya = x(6);
        double yaw = x(9), r = x(11);
        double xa = xc - r * std::sin(yaw);
        double za = zc - r * std::cos(yaw);
        z_pred << xa, za, ya, yaw, r;
        return z_pred;
    }

    Eigen::MatrixXd EKF::computeHJacobian(const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
        double yaw = x(9);
        double r = x(11);

        H(0, 0) = 1;                  // dxa/dxc
        H(0, 9) = -r * std::cos(yaw); // dxa/dyaw
        H(0, 11) = -std::sin(yaw);    // dxa/dr

        H(1, 3) = 1;                 // dza/dzc
        H(1, 9) = r * std::sin(yaw); // dza/dyaw
        H(1, 11) = -std::cos(yaw);   // dza/dr

        H(2, 6) = 1; // dya/dya

        H(3, 9) = 1;  // dyaw/dyaw
        H(4, 11) = 1; // dr/dr

        return H;
    }

} // namespace auto_aim