#include "ekf_singer.hpp"
#include <cmath>

namespace auto_aim
{
    constexpr int STATE_DIM = 9;  // x, vx, ax, y, vy, ay, z, vz, az
    constexpr int MEAS_DIM = 3;   // x, y, z


    EKF_Singer::EKF_Singer()
    {
        x_ = Eigen::VectorXd::Zero(STATE_DIM);
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-1;
        Q_ = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
        R_ = Eigen::MatrixXd::Identity(MEAS_DIM, MEAS_DIM) * 1e-2;

        alpha_ = 0.5;  // Singer 模型衰减因子
    }

    void EKF_Singer::init(double dt)
    {
        dt_ = dt;

        double sigma_a = 2;  // 加速度过程标准差，可调参数
        double q = sigma_a * sigma_a;

        double a = alpha_;
        double e2adt = std::exp(-2 * a * dt_);

        double q11 = (4 * a * a - e2adt * (4 * a * a + 4 * a * a * a * dt_ + a * a * a * a * dt_ * dt_)) / (2 * std::pow(a, 5));
        double q12 = (2 * a - e2adt * (2 * a + 2 * a * a * dt_ + a * a * a * dt_ * dt_)) / (2 * std::pow(a, 4));
        double q13 = (1 - e2adt * (1 + a * dt_ + 0.5 * a * a * dt_ * dt_)) / (2 * std::pow(a, 3));
        double q22 = q13;
        double q23 = (1 - e2adt * (1 + a * dt_)) / (2 * a * a);
        double q33 = (1 - e2adt) / (2 * a);

        Q_ = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

        for (int dim = 0; dim < 3; ++dim)
        {
            int idx = dim * 3;
            Q_(idx, idx)     = q * q11;
            Q_(idx, idx + 1) = q * q12;
            Q_(idx, idx + 2) = q * q13;

            Q_(idx + 1, idx)     = Q_(idx, idx + 1);
            Q_(idx + 1, idx + 1) = q * q22;
            Q_(idx + 1, idx + 2) = q * q23;

            Q_(idx + 2, idx)     = Q_(idx, idx + 2);
            Q_(idx + 2, idx + 1) = Q_(idx + 1, idx + 2);
            Q_(idx + 2, idx + 2) = q * q33;
        }
    }

    void EKF_Singer::setState(const Eigen::VectorXd &x0)
    {
        x_ = x0;
    }

    void EKF_Singer::reset(const Eigen::VectorXd &x0)
    {
        x_ = x0;
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-1;
    }

    Eigen::VectorXd EKF_Singer::predict()
    {
        x_ = f(x_);
        Eigen::MatrixXd F = computeFJacobian(x_);
        P_ = F * P_ * F.transpose() + Q_;
        return x_;
    }
    Eigen::VectorXd EKF_Singer::Onlypredict()
    {
        Eigen::VectorXd predict = f(x_);
        return predict;
    }

    Eigen::VectorXd EKF_Singer::update(const Eigen::VectorXd &z)
    {
        Eigen::VectorXd z_pred = h(x_);
        Eigen::VectorXd y = z - z_pred;

        Eigen::MatrixXd H = computeHJacobian(x_);
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H) * P_;
        return x_;
    }

    Eigen::VectorXd EKF_Singer::f(const Eigen::VectorXd &x)
    {
        Eigen::VectorXd x_pred = x;
        double a = alpha_;
        double e = std::exp(-a * dt_);

        for (int dim = 0; dim < 3; ++dim)
        {
            int idx = dim * 3;
            double p = x(idx);
            double v = x(idx + 1);
            double acc = x(idx + 2);

            x_pred(idx)     = p + (1 - e + a * dt_ * e) * v / a + (dt_ - (1 - e) / a) * acc / a;
            x_pred(idx + 1) = v + (1 - e) * acc / a;
            x_pred(idx + 2) = e * acc;
        }

        return x_pred;
    }


    Eigen::MatrixXd EKF_Singer::computeFJacobian(const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        double a = alpha_;
        double e = std::exp(-a * dt_);

        for (int i = 0; i < 3; ++i)
        {
            int idx = i * 3;

            F(idx, idx + 1) = (1 - e + a * dt_ * e) / a;
            F(idx, idx + 2) = (dt_ - (1 - e) / a) / a;
            F(idx + 1, idx + 2) = (1 - e) / a;
            F(idx + 2, idx + 2) = e;
        }

        return F;
    }

    Eigen::VectorXd EKF_Singer::h(const Eigen::VectorXd &x)
    {
        Eigen::VectorXd z_pred(MEAS_DIM);
        z_pred << x(0), x(3), x(6);  // x, y, z 位置
        return z_pred;
    }

    Eigen::MatrixXd EKF_Singer::computeHJacobian(const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
        H(0, 0) = 1;  // dx/dx
        H(1, 3) = 1;  // dy/dy
        H(2, 6) = 1;  // dz/dz
        return H;
    }

}
