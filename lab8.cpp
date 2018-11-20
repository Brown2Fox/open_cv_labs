#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>


int thresholdValue = 0;
int thresholdType = 3;
int const maxValue = 255;
int const maxType = 4;
int const maxBinaryValue = 255;
cv::Mat src, srcGray;
cv::Mat thresholdImage;
cv::Mat adaptiveThresholdImage;

#define ADAPTIVE_THRESHOLD_WINDOW_TITLE "Adaptive thresholding"
#define THRESHOLD_WINDOW_TITLE "Thresholding"
#define GAMMA_WINDOW_TITLE "Brightness/contrast/gamma adjustments"

void onThresholdChanged(int, void *)
{
    cv::Mat thresMat;
    cv::threshold(srcGray, thresMat, thresholdValue, maxValue, thresholdType);
    cv::imshow(THRESHOLD_WINDOW_TITLE, thresMat);
}

void showTresholding(const cv::Mat &image)
{
    src = image.clone();
    cv::cvtColor(src, srcGray, CV_BGR2GRAY);
    cv::namedWindow(THRESHOLD_WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold type", THRESHOLD_WINDOW_TITLE, &thresholdType, maxType, onThresholdChanged);
    cv::createTrackbar("Threshold", THRESHOLD_WINDOW_TITLE, &thresholdValue, maxValue, onThresholdChanged);
    onThresholdChanged(0, nullptr);
}

int adaptiveMethod = 0;
const int maxAdaptiveMethod = 1;
int blockSize = 3;
int C = 1;
void onAdaptiveThresholdChanged(int, void *)
{
    if (blockSize % 2 == 0)
    {
        blockSize += 1; // should be odd
    }
    if (blockSize <= 3)
    {
        blockSize = 3; // min block size
    }

    cv::Mat adaptiveDst;
    cv::adaptiveThreshold(srcGray, adaptiveDst, 50, adaptiveMethod, cv::THRESH_BINARY, blockSize, C);
    cv::imshow(ADAPTIVE_THRESHOLD_WINDOW_TITLE, adaptiveDst);
}

void showAdaptiveTresholding(const cv::Mat &image)
{
    src = image.clone();
    if (src.rows > 1600 || src.cols > 2560)
    {
        cv::resize(src, src, cv::Size(src.cols / 5, src.rows / 5));
    }
    cv::cvtColor(src, srcGray, CV_BGR2GRAY);
    cv::namedWindow(ADAPTIVE_THRESHOLD_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    cv::createTrackbar("Adaptive method", ADAPTIVE_THRESHOLD_WINDOW_TITLE, &adaptiveMethod, maxAdaptiveMethod, onAdaptiveThresholdChanged);
    cv::createTrackbar("Block size", ADAPTIVE_THRESHOLD_WINDOW_TITLE, &blockSize, maxValue, onAdaptiveThresholdChanged);
    cv::createTrackbar("Const", ADAPTIVE_THRESHOLD_WINDOW_TITLE, &C, maxValue, onAdaptiveThresholdChanged);
    onAdaptiveThresholdChanged(0, nullptr);
}

void drawCont(const cv::Mat &image, const std::string windowToShow)
{
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = image.clone();
    cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    // Отрисовка контуров
    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (size_t idx = 0; idx < contours.size(); idx++)
    {
        cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
    }
    cv::imshow(windowToShow, contourImage);
    cv::moveWindow(windowToShow, 200, 0);
}

int alpha_value = 100;
int beta_value = 0;
// оригинал изображения и изображение с изменненной гаммой/яркостью/контрастностью

cv::Mat img_original, img_corrected;

// Измененение контраста и яркости
void basicLinearTransform(const cv::Mat &inp, cv::Mat &outp)
{
    // Изменить яркость и контрастность
    auto alpha = ((double)alpha_value) / 100;
    auto beta  = (double)beta_value;
    inp.convertTo(outp, 0, alpha, beta);
}

// Обработка изменения контраста
void onContrastChangeListener(int a = 0 , void * b = nullptr)
{
    basicLinearTransform(img_original, img_corrected);
    cv::imshow("Brightness/contrast/gamma adjustments", img_corrected);
}

// Обработка изменения яркости
void onBrightnessChangeListener(int a = 0, void * b = nullptr)
{
    basicLinearTransform(img_original, img_corrected);
    cv::imshow("Brightness/contrast/gamma adjustments", img_corrected);
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        throw std::invalid_argument("Wrong number of parameters!");
    }
    img_original = cv::imread(argv[1]);
    if (!img_original.data)
    {
        throw std::invalid_argument("No image data found!");
    }
    img_corrected = cv::imread(argv[1]);

    /* Создание окна изменения яркости/гаммы/контраста */



    cv::namedWindow(GAMMA_WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Contrast", GAMMA_WINDOW_TITLE, &alpha_value, 1000, onContrastChangeListener);
    cv::createTrackbar("Brightness", GAMMA_WINDOW_TITLE, &beta_value, 1000, onBrightnessChangeListener);
    onContrastChangeListener();
    onBrightnessChangeListener();

    cv::waitKey(0);
    cv::destroyWindow(GAMMA_WINDOW_TITLE);

    // Бинаризация изображения
    showTresholding(img_corrected);
    showAdaptiveTresholding(img_corrected);
    cv::waitKey(0);
    cv::destroyWindow(ADAPTIVE_THRESHOLD_WINDOW_TITLE);
    cv::destroyWindow(THRESHOLD_WINDOW_TITLE);

    // Отрисовка границ
    drawCont(thresholdImage, "Contours from threshold image");
    drawCont(adaptiveThresholdImage, "Contours from adaptive threshold image");
    cv::waitKey(0);
    return 0;
}
