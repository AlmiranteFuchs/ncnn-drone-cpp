#include "yolo-fastestv2.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    static const char* class_names[] = {
        "black box", "blue box", "green box", "red box", "white box"
    };


    yoloFastestv2 api;



    api.loadModel("./model/yolo-fastestv2-opt.param", "./model/yolo-fastestv2-opt.bin");

    // Define input and output directories
    fs::path inputDir = "../../../test";
    fs::path outputDir = "./output_images";

    // Create output directory if it does not exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // Iterate over files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            std::string inputFile = entry.path().string();
            std::string outputFile = (outputDir / entry.path().filename()).string();

            // Read image
            cv::Mat cvImg = cv::imread(inputFile);
            if (cvImg.empty()) {
                std::cerr << "Error reading image: " << inputFile << std::endl;
                continue;
            }

            // Perform detection
            std::vector<TargetBox> boxes;
            api.detection(cvImg, boxes);

            // Draw bounding boxes and labels
            for (const auto& box : boxes) {
                std::cout << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2
                          << " " << box.score << " " << box.cate << std::endl;


                if (box.cate < 0 || box.cate >= sizeof(class_names) / sizeof(class_names[0])) {
    			std::cerr << "Invalid category index: " << box.cate << std::endl;
    			continue; // Skip this box
		}



                char text[256];
                sprintf(text, "%s %.1f%%", class_names[box.cate], box.score * 100);
                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int x = box.x1;
                int y = box.y1 - label_size.height - baseLine;
                if (y < 0) y = 0;
                if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

                cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(255, 255, 255), -1);
                cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cv::rectangle(cvImg, cv::Point(box.x1, box.y1),
                              cv::Point(box.x2, box.y2), cv::Scalar(255, 255, 0), 2, 2, 0);
            }

            // Save output image
            if (!cv::imwrite(outputFile, cvImg)) {
                std::cerr << "Error writing image: " << outputFile << std::endl;
            }
        }
    }

    return 0;
}

