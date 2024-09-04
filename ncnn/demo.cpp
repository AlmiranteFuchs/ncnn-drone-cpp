#include "yolo-fastestv2.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    static const char* class_names[] = {
        "black box", "blue box", "green box", "red box", "white box"
    };

    // Load the YOLO model
    yoloFastestv2 api;
    api.loadModel("./model/drone-yolo-fastestv2.param", "./model/drone-yolo-fastestv2.bin");

    // Open the Raspberry Pi camera
    cv::VideoCapture cap(1);  // 0 is usually the default camera index
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (true) {
        // Capture frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error capturing frame!" << std::endl;
            break;
        }

        // Resize the image to 416x416
        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(416, 416));

        // Perform detection
        std::vector<TargetBox> boxes;
        api.detection(resizedFrame, boxes);

        // Draw bounding boxes and labels
        for (const auto& box : boxes) {
            if (box.cate < 0 || box.cate >= sizeof(class_names) / sizeof(class_names[0])) {
                std::cerr << "Invalid category index: " << box.cate << std::endl;
                continue;  // Skip this box
            }

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[box.cate], box.score * 100);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int x = box.x1;
            int y = box.y1 - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > resizedFrame.cols) x = resizedFrame.cols - label_size.width;

            // Draw the bounding box and label
            cv::rectangle(resizedFrame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);
            cv::putText(resizedFrame, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            cv::rectangle(resizedFrame, cv::Point(box.x1, box.y1),
                          cv::Point(box.x2, box.y2), cv::Scalar(255, 255, 0), 2, 2, 0);
        }

        // Calculate FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - startTime;
        if (elapsed.count() >= 1.0) {  // Update FPS every second
            float fps = frameCount / elapsed.count();
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(resizedFrame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            frameCount = 0;
            startTime = std::chrono::high_resolution_clock::now();
        }

        // Show the output with bounding boxes and FPS
        cv::imshow("YOLO Detection", resizedFrame);

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
