#include "detector.hpp"
#include "img_tools.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include <opencv2/opencv.hpp>
#include "armor.hpp"
#include "PnP.hpp"
#include "ukf.hpp"
#include "plot.hpp"
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <fstream>    // 添加文件输出流
#include <iomanip>    // 添加格式化输出

#define VIDEO "/home/wxy/NJU_RMCV/test/video/linear_video/linear_difficult.avi"

using namespace auto_aim;

// 全局轨迹缓存和可视化器
auto_aim::TrajectoryBuffer trajectory_buffer;
auto_aim::TrajectoryVisualizer trajectory_visualizer(500, 500);

//  相机内参
static const cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) <<  1286.307063384126 , 0                  , 645.34450819155256, 
                                0                 , 1288.1400736562441 , 483.6163720308021 , 
                                0                 , 0                  , 1                   );
// 畸变系数
static const cv::Mat distort_coeffs =
    (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
// clang-format on

// 定义装甲板信息结构体，用于面板显示
struct ArmorInfo {
    std::string color;
    std::string name;
    float confidence;
    cv::Point2f position;
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw, pitch, roll;
    cv::Scalar color_value;

    // 新增：UKF平滑后的三维位置、速度、加速度、yaw等
    Eigen::Matrix<double, UKF::n_x, 1> ukf_state;
    bool has_ukf = false;
};

// 线程安全的帧缓冲
class FrameQueue {
public:
    void push(const cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(frame.clone()); // 复制帧，确保线程安全
        lock.unlock();
        cond_.notify_one();
    }

    bool pop(cv::Mat& frame, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            // 等待新帧或超时
            auto result = cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                                       [this] { return !queue_.empty() || stop_; });
            if (!result || stop_) return false;
        }
        
        frame = queue_.front();
        queue_.pop();
        return true;
    }
    
    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        lock.unlock();
        cond_.notify_all();
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        std::queue<cv::Mat> empty;
        std::swap(queue_, empty);
    }

private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool stop_ = false;
};

// 线程安全的装甲板结果缓冲
class ResultBuffer {
public:
    void update(const cv::Mat& img, const std::vector<ArmorInfo>& armors) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!img.empty()) {
            result_image_ = img.clone();
        }
        armor_info_ = armors;
        updated_ = true;
        lock.unlock();
    }

    bool get(cv::Mat& img, std::vector<ArmorInfo>& armors) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!updated_ || result_image_.empty()) {
            return false;
        }
        img = result_image_.clone();
        armors = armor_info_;
        return true;
    }

private:
    cv::Mat result_image_;
    std::vector<ArmorInfo> armor_info_;
    std::mutex mutex_;
    bool updated_ = false;
};

// 创建信息面板
void createInfoPanel(cv::Mat& panel, const std::vector<ArmorInfo>& armors) {
    panel = cv::Mat(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(panel, "Armor Detection Info Panel", 
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 
                0.8, cv::Scalar(0, 0, 0), 2);
    cv::line(panel, cv::Point(0, 40), cv::Point(600, 40), 
             cv::Scalar(0, 0, 0), 2);

    const int infoHeight = 80;
    const int startY = 60;
    int maxArmors = std::min(4, static_cast<int>(armors.size()));

    for (int i = 0; i < maxArmors; i++) {
        const auto& armor = armors[i];
        int y = startY + i * infoHeight;

        cv::rectangle(panel, 
                      cv::Point(10, y - 15), 
                      cv::Point(590, y + infoHeight - 25), 
                      cv::Scalar(240, 240, 240), 
                      cv::FILLED);

        cv::putText(panel, 
                    fmt::format("Armor #{}: {} {}", i+1, armor.color, armor.name), 
                    cv::Point(20, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, armor.color_value, 2);

        cv::putText(panel, 
                    fmt::format("Conf: {:.2f}", armor.confidence), 
                    cv::Point(320, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 0), 1);

        // 只显示UKF平滑后的三维位置和姿态
        if (armor.has_ukf) {
            cv::putText(panel, 
                fmt::format("UKF Pos: x {:.2f} y {:.2f} z {:.2f}", 
                    armor.ukf_state(0), armor.ukf_state(6), armor.ukf_state(3)),
                cv::Point(20, y + 30), 
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, cv::Scalar(0, 128, 255), 1);

            cv::putText(panel, 
                fmt::format("UKF Yaw(deg): {:.2f}  R: {:.3f}", 
                    armor.ukf_state(9) * 180.0 / CV_PI, armor.ukf_state(11)),
                cv::Point(20, y + 55), 
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, cv::Scalar(0, 128, 255), 1);
        } else {
            cv::putText(panel, "No UKF data", cv::Point(20, y + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }

    if (armors.empty()) {
        cv::putText(panel, "No armor detected", 
                    cv::Point(150, 200), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1, cv::Scalar(0, 0, 255), 2);
    }
}

// 装甲板检测线程函数
void detectionThread(FrameQueue& input_queue, 
                    ResultBuffer& result_buffer, 
                    std::atomic<bool>& running,
                    Detector& detector,
                    PNPSolver& pnp_solver) {
    cv::Mat frame;

    // 新增：为每个装甲板维护一个UKF
    std::map<std::string, UKF> ukf_map;
    std::map<std::string, double> last_time_map;
    
    // 新增：存储上一帧的预测值，用于误差分析
    std::map<std::string, cv::Point3f> last_predict_pos;
    std::map<std::string, double> last_predict_yaw;
    
    double fps = 60.0;
    double dt = 1.0 / fps;
    
    // 创建CSV文件并写入表头
    std::ofstream csv_file("/home/wxy/NJU_RMCV/prediction_error.csv");
    csv_file << "res_x,res_y,res_z,res_yaw" << std::endl;
    
    int frame_id = 0;

    while (running) {
        // 获取帧，如果队列为空则等待
        if (!input_queue.pop(frame)) {
            if (!running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        frame_id++;
        
        // 检测装甲板
        auto armors = detector.detect(frame);

        // 创建副本用于绘制
        cv::Mat draw_img = frame.clone();
        std::vector<ArmorInfo> armorInfoList;
        
        // 处理每个装甲板
        for (const auto& armor : armors) {
            // 绘制装甲板点
            tools::draw_points(draw_img, armor.points);
            
            // 显示装甲板标识
            tools::draw_text(
                draw_img,
                fmt::format("{},{},{:.2f}", COLORS[armor.color], ARMOR_NAMES[armor.name], armor.confidence),
                armor.left.top
            );

            // 准备图像点
            std::vector<cv::Point2f> img_points{
                armor.left.top,
                armor.right.top,
                armor.right.bottom,
                armor.left.bottom
            };

            // PNP解算
            cv::Mat rvec, tvec;
            double yaw_val;
            pnp_solver.solvePose(img_points, rvec, tvec, yaw_val);
            
            // 获取欧拉角
            double pitch, roll;
            pnp_solver.getEulerAngles(rvec, yaw_val, pitch, roll);
            
            // 确定装甲板颜色
            cv::Scalar color_value;
            if (COLORS[armor.color] == "BLUE") {
                color_value = cv::Scalar(255, 0, 0);
            } else if (COLORS[armor.color] == "RED") {
                color_value = cv::Scalar(0, 0, 255);
            } else {
                color_value = cv::Scalar(0, 128, 0);
            }
            
            // 生成唯一ID
            std::string armor_id = fmt::format("{}_{}", COLORS[armor.color], ARMOR_NAMES[armor.name]);

            // 当前帧的实际观测值
            cv::Point3f actual_pos(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            double actual_yaw = yaw_val;

            // 误差分析：计算实际值与上一帧预测值的差
            if (last_predict_pos.find(armor_id) != last_predict_pos.end() && frame_id > 1) {
                cv::Point3f last_pred = last_predict_pos[armor_id];
                double last_pred_yaw = last_predict_yaw[armor_id];
                
                // 计算误差
                double res_x = actual_pos.x - last_pred.x;
                double res_y = actual_pos.y - last_pred.y;
                double res_z = actual_pos.z - last_pred.z;
                double res_yaw = actual_yaw - last_pred_yaw;
                
                // 角度误差标准化到 [-π, π]
                while (res_yaw > M_PI) res_yaw -= 2 * M_PI;
                while (res_yaw < -M_PI) res_yaw += 2 * M_PI;
                
                // 写入CSV文件
                csv_file << std::fixed << std::setprecision(6)
                    << res_x << ","
                    << res_y << ","
                    << res_z << ","
                    << res_yaw << std::endl;
                
                // 在控制台输出误差信息（可选）
                std::cout << fmt::format("Frame {}: {} - Error: x={:.3f}, y={:.3f}, z={:.3f}, yaw={:.3f}°",
                    frame_id, armor_id, res_x, res_y, res_z, res_yaw * 180.0 / M_PI) << std::endl;
            }

            // 观测向量zk = [xa, za, ya, yaw, r]
            UKF::VectorZ z;
            double xa = tvec.at<double>(0);
            double za = tvec.at<double>(2);
            double ya = tvec.at<double>(1);
            double r_val = 0.13 / 2;

            z << xa, za, ya, yaw_val, r_val;

            // UKF初始化
            if (ukf_map.find(armor_id) == ukf_map.end()) {
                UKF ukf;
                UKF::VectorX x0 = UKF::VectorX::Zero();
                x0(0) = xa; // xc
                x0(3) = za; // zc
                x0(6) = ya; // yc
                x0(9) = yaw_val; // yaw
                x0(11) = r_val; // r
                ukf.init(x0, UKF::MatrixX::Identity());
                ukf_map[armor_id] = ukf;
                last_time_map[armor_id] = 0;
            }

            // UKF predict & update
            auto& ukf = ukf_map[armor_id];
            ukf.predict(dt);
            ukf.update(z);

            // 获取UKF状态并预测下一帧位置
            UKF::VectorX current_state = ukf.getState();
            
            // 预测下一帧的位置（用于下一次误差计算）
            cv::Point3f next_predict_pos = ukf.predictPosition(dt);
            double next_predict_yaw = current_state(9) + current_state(10) * dt; // yaw + omega * dt
            
            // 存储预测值供下一帧使用
            last_predict_pos[armor_id] = next_predict_pos;
            last_predict_yaw[armor_id] = next_predict_yaw;

            // 保存UKF状态到ArmorInfo
            ArmorInfo info;
            info.color = COLORS[armor.color];
            info.name = ARMOR_NAMES[armor.name];
            info.confidence = armor.confidence;
            info.position = (armor.left.top + armor.right.bottom) * 0.5;
            info.rvec = rvec.clone();
            info.tvec = tvec.clone();
            info.yaw = yaw_val;
            info.pitch = pitch;
            info.roll = roll;
            info.color_value = color_value;
            info.ukf_state = current_state;
            info.has_ukf = true;
            
            // 添加轨迹点
            if (info.has_ukf) {
                trajectory_buffer.addPoint(armor_id, cv::Point3f(
                    info.ukf_state(0),  // xc
                    info.ukf_state(6),  // yc
                    info.ukf_state(3)   // zc
                ));
            } else {
                trajectory_buffer.addPoint(armor_id, cv::Point3f(
                    tvec.at<double>(0),
                    tvec.at<double>(1),
                    tvec.at<double>(2)
                ));
            }
            
            armorInfoList.push_back(info);
        }
        
        // 更新结果缓冲
        result_buffer.update(draw_img, armorInfoList);
    }
    
    // 关闭CSV文件
    csv_file.close();
    std::cout << "误差分析数据已保存到: /home/wxy/NJU_RMCV/prediction_error.csv" << std::endl;
}

int main(int argc, char *argv[])
{
    auto_aim::Detector detector;
    auto_aim::PNPSolver pnp_solver(camera_matrix, distort_coeffs);
    
    // 打开视频文件
    cv::VideoCapture cap(VIDEO);
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开视频文件！" << std::endl;
        return -1;
    }
    
    // 创建线程间通信的数据结构
    FrameQueue frame_queue;
    ResultBuffer result_buffer;
    std::atomic<bool> running(true);
    
    // 启动装甲板检测线程
    std::thread detect_thread(detectionThread, 
                             std::ref(frame_queue), 
                             std::ref(result_buffer), 
                             std::ref(running), 
                             std::ref(detector), 
                             std::ref(pnp_solver));
    
    // 预分配内存
    cv::Mat frame, result_img, small_img, infoPanel;
    std::vector<ArmorInfo> armorInfoList;
    
    // 用于控制处理速度
    int frame_count = 0;
    const int process_every_n_frames = 1;  // 可调整为2或更大的值
    
    // 主循环：读取视频并显示结果
    while (running) {
        // 读取新帧
        cap >> frame;
        if (frame.empty()) {
            // 视频结束，重新播放
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        // 每N帧处理一次
        frame_count++;
        if (frame_count % process_every_n_frames == 0) {
            // 推送帧到队列中等待处理
            frame_queue.push(frame);
        }
        
        // 获取最新的处理结果
        bool have_result = result_buffer.get(result_img, armorInfoList);
        
        // 显示结果
        if (have_result) {
            // 创建信息面板
            createInfoPanel(infoPanel, armorInfoList);
            
            // 转换ArmorInfo到LineInfo
            std::vector<auto_aim::LineInfo> lineInfoList;
            for (const auto& armor : armorInfoList) {
                auto_aim::LineInfo line;
                line.color = armor.color;
                line.name = armor.name;
                line.confidence = armor.confidence;
                line.position = armor.position;
                line.rvec = armor.rvec.clone();  // 注意需要clone矩阵
                line.tvec = armor.tvec.clone();
                line.yaw = armor.yaw;
                line.pitch = armor.pitch;
                line.roll = armor.roll;
                line.color_value = armor.color_value;
                
                // 复制UKF状态
                if (armor.has_ukf) {
                    line.ukf_state = armor.ukf_state;
                    line.has_ukf = true;
                }
                
                lineInfoList.push_back(line);
            }
            
            // 创建轨迹面板
            cv::Mat trajectoryPanel = trajectory_visualizer.createTrajectoryPanel(
                trajectory_buffer, lineInfoList);
            
            // 缩小图像
            cv::resize(result_img, small_img, cv::Size(960, 540), 0, 0, cv::INTER_NEAREST);
            
            // 显示所有面板
            cv::imshow("Armor Detection (press q to quit)", small_img);
            cv::imshow("Armor Info Panel", infoPanel);
            cv::imshow("Trajectory Visualization", trajectoryPanel);
        }
        
        // 处理键盘事件
        int key = cv::waitKey(1);
        if (key == 'q') {
            running = false;
        } else if (key == 's') {
            // 保存当前帧
            cv::imwrite("screenshot.jpg", frame);
        }
    }
    
    // 在程序结束前添加统计信息
    std::cout << "\n=== 程序运行完成 ===" << std::endl;
    std::cout << "误差分析文件已生成: /home/wxy/NJU_RMCV/prediction_error.csv" << std::endl;
    std::cout << "可以使用 Python 脚本 visualize_bias.py 来分析预测误差" << std::endl;
    
    // 清理资源
    frame_queue.stop();
    trajectory_buffer.clear();
    if (detect_thread.joinable()) {
        detect_thread.join();
    }
    
    cv::destroyAllWindows();
    return 0;
}