//#include "Windows.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CVUI_IMPLEMENTATION

#include <cvui.h>

// For cvui #define CVUI_IMPLEMENTATION #include "cvui.h"

#define MODIFIED_NAME "Modified image"
#define SETTINGS_NAME "Settings"

void replacePixels(const cv::Mat &source, cv::Mat &dest, const uint8_t(&lowerBound)[3], const uint8_t(&upperBound)[3]);

std::string getSourcePath();

int main()
{
    cv::VideoCapture cameraVideoStream(0);// Camera ID 0
    if (!cameraVideoStream.isOpened()) // Exit if there is no device with this id
        return 1;

    cv::Mat cameraFrame; // Frame
    cv::Mat replacer;

    /***************Image window init*******************/
    cvui::init(MODIFIED_NAME);
    /*******************************************************/

    /***************Settings window init*******************/
    cv::Mat settingsFrame;
    settingsFrame.create(475, 200, 0);
    cvui::init(SETTINGS_NAME);
    /*******************************************************/

    /***************Pixel Replacing Settings init******************/
    bool useReplace = false;
    uint8_t lowerBound[3]{0, 0, 0};
    uint8_t upperBound[3]{255, 255, 255};
    /*******************************************************/

    while (true)
    {
        cameraFrame.release();
        cameraVideoStream.read(cameraFrame);

        /***************Settings window draw*******************/
        cvui::window(settingsFrame, 0, 0, 200, 475, "Pixel Replacing");     // Checkbar for pixel replacing
        cvui::checkbox(settingsFrame, 0, 25, "Use Replace", &useReplace);
        if (!replacer.data)
        {
            cvui::text(settingsFrame, 100, 27, "Needs picture", 0.4, 0xFFCEFF);
            useReplace = false;
        }

        // Button for open file dialog
        if (cvui::button(settingsFrame, 30, 50, "Choose source"))
        {
            std::string fileName = getSourcePath();       // This realization works only with pictures, not video
            replacer = cv::imread(fileName);
            if (!replacer.data)
            {
                std::cerr << "Can't open source file";
                exit(1);
            }
            resize(replacer, replacer, cv::Size(cameraFrame.cols, cameraFrame.rows), 0, 0, cv::INTER_CUBIC);
        }

        // Trackbar start position
        int trackBarsY = 95;
        cvui::window(settingsFrame, 0, trackBarsY, 200, 190, "Lower bound");
        cvui::text(settingsFrame, 10, trackBarsY + 50, "b");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 35, 165, &lowerBound[0], 0, 255);
        cvui::text(settingsFrame, 10, trackBarsY + 100, "g");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 85, 165, &lowerBound[1], 0, 255);
        cvui::text(settingsFrame, 10, trackBarsY + 150, "r");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 135, 165, &lowerBound[2], 0, 255);

        cvui::window(settingsFrame, 0, trackBarsY + 190, 200, 190, "Upper bound");
        cvui::text(settingsFrame, 10, trackBarsY + 240, "b");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 225, 165, &upperBound[0], 0, 255);
        cvui::text(settingsFrame, 10, trackBarsY + 290, "g");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 275, 165, &upperBound[1], 0, 255);
        cvui::text(settingsFrame, 10, trackBarsY + 340, "r");
        cvui::trackbar<uint8_t>(settingsFrame, 20, trackBarsY + 325, 165, &upperBound[2], 0, 255);

        cvui::update();
        cv::imshow(SETTINGS_NAME, settingsFrame);     /*******************************************************/

        if (useReplace)
        {
            replacePixels(replacer, cameraFrame, lowerBound, upperBound);
        }
        cv::imshow(MODIFIED_NAME, cameraFrame);

        char c = cvWaitKey(33);
        if (c == 27) break; // Exit if pressed Esc

    }
    cameraFrame.release();
    replacer.release();
    return 0;
}

std::string getSourcePath()
{
    return "pics/Render.png";
}

void replacePixels(const cv::Mat &source, cv::Mat &dest, const uint8_t(&lowerBound)[3], const uint8_t(&upperBound)[3])
{
    uint8_t *pixelPtr = dest.data;
    int destCn = dest.channels();
    int sourceCn = source.channels();

    for (int i = 0; i < dest.rows; i++)
    {
        for (int j = 0; j < dest.cols; j++)
        {
            uint8_t b = pixelPtr[i * dest.cols * destCn + j * destCn + 0];
            uint8_t g = pixelPtr[i * dest.cols * destCn + j * destCn + 1];
            uint8_t r = pixelPtr[i * dest.cols * destCn + j * destCn + 2];

            if (b >= lowerBound[0] && b <= upperBound[0] && g >= lowerBound[1] && g <= upperBound[1] && r >= lowerBound[2] && r <= upperBound[2])
            {
                for (int k = 0; k < 3; k++)
                {
                    pixelPtr[i * dest.cols * destCn + j * destCn + k] = source.data[i * source.cols * sourceCn + j * sourceCn + k];
                }
            }
        }
    }
}