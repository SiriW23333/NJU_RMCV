#include "detector.hpp"
#include "img_tools.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include <opencv2/opencv.hpp>
#include "armor.hpp"
#include "tracker.hpp"
#include "PnP.hpp"
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

using namespace auto_aim;

// clang-format off
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
    cv::Point2f position;  // 装甲板中心位置
    cv::Mat rvec;          // 旋转向量
    cv::Mat tvec;          // 平移向量
    double yaw, pitch, roll;  // 欧拉角
    cv::Scalar color_value;   // 用于显示的颜色
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
    // 创建白色背景面板
    panel = cv::Mat(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 画标题
    cv::putText(panel, "Armor Detection Info Panel", 
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 
                0.8, cv::Scalar(0, 0, 0), 2);
    
    // 画分隔线
    cv::line(panel, cv::Point(0, 40), cv::Point(600, 40), 
             cv::Scalar(0, 0, 0), 2);
    
    // 设置每个装甲板信息区域的高度
    const int infoHeight = 80;
    const int startY = 60;
    
    // 最多显示4个装甲板信息
    int maxArmors = std::min(4, static_cast<int>(armors.size()));
    
    for (int i = 0; i < maxArmors; i++) {
        const auto& armor = armors[i];
        int y = startY + i * infoHeight;
        
        // 为每个装甲板绘制背景色块
        cv::rectangle(panel, 
                      cv::Point(10, y - 15), 
                      cv::Point(590, y + infoHeight - 25), 
                      cv::Scalar(240, 240, 240), 
                      cv::FILLED);
        
        // 绘制装甲板编号和颜色信息
        cv::putText(panel, 
                    fmt::format("Armor #{}: {} {}", i+1, armor.color, armor.name), 
                    cv::Point(20, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, armor.color_value, 2);
        
        // 绘制置信度
        cv::putText(panel, 
                    fmt::format("Conf: {:.2f}", armor.confidence), 
                    cv::Point(320, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 0), 1);
        
        // 绘制位置信息
        cv::putText(panel, 
                    fmt::format("Pos: x {:.2f} y {:.2f} z {:.2f}", 
                                armor.tvec.at<double>(0), 
                                armor.tvec.at<double>(1), 
                                armor.tvec.at<double>(2)), 
                    cv::Point(20, y + 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 0), 1);
        
        // 绘制姿态信息
        cv::putText(panel, 
                    fmt::format("Pose: yaw {:.2f} pitch {:.2f} roll {:.2f}", 
                                armor.yaw * 180 / CV_PI, 
                                armor.pitch * 180 / CV_PI, 
                                armor.roll * 180 / CV_PI), 
                    cv::Point(20, y + 55), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // 如果没有检测到装甲板
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
    
    while (running) {
        // 获取帧，如果队列为空则等待
        if (!input_queue.pop(frame)) {
            if (!running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
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
            pnp_solver.solvePose(img_points, rvec, tvec);
            
            // 获取欧拉角
            double yaw, pitch, roll;
            pnp_solver.getEulerAngles(rvec, yaw, pitch, roll);
            
            // 确定装甲板颜色
            cv::Scalar color_value;
            if (COLORS[armor.color] == "BLUE") {
                color_value = cv::Scalar(255, 0, 0);
            } else if (COLORS[armor.color] == "RED") {
                color_value = cv::Scalar(0, 0, 255);
            } else {
                color_value = cv::Scalar(0, 128, 0);
            }
            
            // 保存装甲板信息
            ArmorInfo info;
            info.color = COLORS[armor.color];
            info.name = ARMOR_NAMES[armor.name];
            info.confidence = armor.confidence;
            info.position = (armor.left.top + armor.right.bottom) * 0.5;
            info.rvec = rvec.clone();
            info.tvec = tvec.clone();
            info.yaw = yaw;
            info.pitch = pitch;
            info.roll = roll;
            info.color_value = color_value;
            
            armorInfoList.push_back(info);
        }
        
        // 更新结果缓冲
        result_buffer.update(draw_img, armorInfoList);
    }
}

int main(int argc, char *argv[])
{
    auto_aim::Detector detector;
    auto_aim::PNPSolver pnp_solver(camera_matrix, distort_coeffs);
    
    // 打开视频文件
    cv::VideoCapture cap("/home/wxy/NJU_RMCV/src/spin_staight.mp4");
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
            
            // 缩小图像
            cv::resize(result_img, small_img, cv::Size(960, 540), 0, 0, cv::INTER_NEAREST);
            
            // 显示画面和信息面板
            cv::imshow("Armor Detection (press q to quit)", small_img);
            cv::imshow("Armor Info Panel", infoPanel);
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
    
    // 清理资源
    frame_queue.stop();
    if (detect_thread.joinable()) {
        detect_thread.join();
    }
    
    cv::destroyAllWindows();
    return 0;
}