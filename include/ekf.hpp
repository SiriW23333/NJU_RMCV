// ekf.hpp
#ifndef AUTO_AIM__EKF_HPP
#define AUTO_AIM__EKF_HPP

#include <Eigen/Dense>

namespace auto_aim
{

    class EKF
    {
    public:
        EKF();

        void init(double dt);
        void setState(const Eigen::VectorXd &x0);
        void reset(const Eigen::VectorXd &x0);

        Eigen::VectorXd predict();
        Eigen::VectorXd update(const Eigen::VectorXd &z);
        Eigen::VectorXd getState() const;
        Eigen::VectorXd Onlypredict();
    
    private:
        double dt_;

        // 状态变量
        Eigen::VectorXd x_; // 10x1
        Eigen::MatrixXd P_; // 状态协方差矩阵 10x10
        Eigen::MatrixXd Q_; // 过程噪声协方差矩阵 10x10
        Eigen::MatrixXd R_; // 观测噪声协方差矩阵 5x5

        // 状态转移函数及其 Jacobian
        Eigen::VectorXd f(const Eigen::VectorXd &x);
        Eigen::MatrixXd computeFJacobian(const Eigen::VectorXd &x);

        // 观测函数及其 Jacobian
        Eigen::VectorXd h(const Eigen::VectorXd &x);
        Eigen::MatrixXd computeHJacobian(const Eigen::VectorXd &x);
    };

} // namespace auto_aim

#endif // AUTO_AIM__EKF_HPP
