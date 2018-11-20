#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

using Points2D = std::vector<std::vector<cv::Point>>;

Mat markerMask, img;
Point prevPt(-1, -1);

namespace KmeansData
{
    const unsigned int SLIDER_MAX = 25;
    int MAX_ITERATIONS = 2;
    int sliderValue;
    Mat image;
}

Mat applyKMeans(const Mat& source){
    if (KmeansData::sliderValue==0){
        KmeansData::sliderValue=1;
    }

    std::chrono::high_resolution_clock::time_point start,end; // for time measurement
    std::chrono::duration<double> elapsedTime{};			 	  // for time measurement
    start = std::chrono::high_resolution_clock::now();
    const auto singleLineSize = static_cast<const unsigned int>(source.rows * source.cols);
    Mat data = source.reshape(1, singleLineSize);
    data.convertTo(data, CV_32F);
    std::vector<int> labels;
    cv::Mat1f colors;

    cv::kmeans(data,
               KmeansData::sliderValue,
               labels,
               cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),
               KmeansData::MAX_ITERATIONS,
               cv::KMEANS_PP_CENTERS,
               colors);

    for (unsigned int i = 0 ; i < singleLineSize ; i++ )
    {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }

    Mat outputImage = data.reshape(3, source.rows);
    outputImage.convertTo(outputImage, CV_8U);
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout
            << "Elapsed time for processing with "
               << KmeansData::sliderValue
            <<" clusters "
              << elapsedTime.count()
            << "s"
               << std::endl;
    return outputImage;
}

/*****************************************************************************************************************************/

void trackBarMovement(int, void*){
    cv::imshow("KMeans demo", applyKMeans(KmeansData::image));
}



void onMouse(int event, int x, int y, int flags, void *)
{
    if (y < 0 || y >= img.rows) return;
    if (x < 0 || x >= img.cols) return;


    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
    {
        prevPt = Point(1, -1);
    }
    else if (event == EVENT_LBUTTONDOWN)
    {
        prevPt = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0) prevPt = pt;
        line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
        line(img, prevPt, pt, Scalar::all(255), 5, 8, 0);
        prevPt = pt;
        cv::imshow("image", img);
    }
}


void AutoMode(Mat& image)
{
    resize(image, image, Size(700, 500));
    cv::imshow("image", image);

    Mat imageGray, imageBin;

    cv::cvtColor(image, imageGray, CV_BGR2GRAY);
    cv::threshold(imageGray, imageGray, 130, 250, THRESH_BINARY);
    int compCount = 0;
    Points2D contours;
    vector<Vec4i> hierarchy;

    findContours(imageGray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return;

    Mat markers(imageGray.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;
    for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
    {
        drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
    }

    if (compCount == 0) return;

    std::vector<Vec3b> colorTab(static_cast<unsigned long long int>(compCount));

    for (int i = 0; i < compCount; i++)
    {
        colorTab[i] = Vec3b(static_cast<uchar>(rand() & 255), static_cast<uchar>(rand() & 255), static_cast<uchar>(rand() & 255));
    }

    cv::watershed(image, markers);
    Mat wshed(markers.size(), CV_8UC3);

    for (int i = 0; i < markers.rows; i++)
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);

            if (index == -1)
            {
                wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            } else if (index == 0)
            {
                wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            } else
            {
                wshed.at<Vec3b>(i, j) = colorTab[index - 1];
            }
        }

    cv::imshow("watershed transform", wshed);

}

void ManualMode(Mat& img0, Mat& imgGray)
{
    int i, j, compCount = 0;
    Points2D contours;
    vector<Vec4i> hierarchy;

    findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return;

    Mat markers(markerMask.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;

    for (; idx >= 0; idx = hierarchy[idx][0], compCount++) drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
    if (compCount == 0) return;

    vector<Vec3b> colorTab;
    for (i = 0; i < compCount; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colorTab.emplace_back((uchar) b, (uchar) g, (uchar) r);
    }

    auto t = (double) getTickCount();
    cv::watershed(img0, markers);
    t = (double) getTickCount() - t;
    cout << "execution time = " << t * 1000. / getTickFrequency() << "\n";

    Mat wshed(markers.size(), CV_8UC3);

    for (i = 0; i < markers.rows; i++)
        for (j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index == -1) wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            else if (index <= 0 || index > compCount) wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else wshed.at<Vec3b>(i, j) = colorTab[index - 1];
        }
    wshed = wshed * 0.5 + imgGray * 0.5;
    cv::imshow("watershed transform", wshed);
}

int main(int argc, char *argv[])
{
    cout << "METHODS OF USING WATERSHED SEGMENTATION\n";
    cout << "Enter mode: a  --(auto) | h  --(hand) \n";
    char s;
    cin >> s;

    if (s == 'a')
    {
        cout << "Enter type: p --(picture) | v --(video) | c --(camera) \n";
        cin >> s;

        if (s == 'v')
        {
            cout << "Name of video: \n";
            string filename;
            cin >> filename;

            VideoCapture cap("vids/" + filename);

            while (cap.isOpened())
            {
                Mat image;
                cap >> image;
                AutoMode(image);

                auto key = waitKey(100);

                if (key == 27) return 0;
            }
        }
        else if (s == 'c')
        {
            VideoCapture cap(0);

            while (cap.isOpened())
            {
                Mat image;
                cap >> image;
                AutoMode(image);

                auto key = waitKey(100);

                if (key == 27) return 0;
            }
        }
        else if (s == 'p')
        {
            cout << "Name of picture: \n";
            string filename;
            cin >> filename;

            Mat image;
            image = imread("pics/" + filename, CV_LOAD_IMAGE_COLOR);

            while(1)
            {
                AutoMode(image);

                auto key = waitKey(100);

                if (key == 27) return 0;
            }
        }
        else
        {
            cout << "Incorrect parameters";
        }
    }
    else if (s == 'h')
    {
        cout << "Enter type: p --(picture) | v --(video) | c --(cam)\n";
        cin >> s;
        cout.clear();
        cout    << "\nHELP\n"
                << "--move pressed left mouse button to mark picture\n"
                << "--press R to reset the mask\n"
                << "--press Space to apply\n\n";

        if (s == 'p')
        {
            cout << "Name of image: \n";
            string filename;
            cin >> filename;
            Mat img0 = imread("pics/" + filename, 1), imgGray;
            resize(img0, img0, Size(900, 1600));
            cv::imshow("image", img0);

            img0.copyTo(img);
            cv::cvtColor(img, markerMask, COLOR_BGR2GRAY);
            cv::cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);

            markerMask = Scalar::all(0);

            setMouseCallback("image", onMouse, 0);

            while (true)
            {
                auto c = waitKey(0);

                if (c == 'e') return 0;

                if (c == 27) break;

                if (c == 'r')
                {
                    markerMask = Scalar::all(0);
                    img0.copyTo(img);
                    cv::imshow("image", img);
                }

                if (c == ' ')
                {
                    ManualMode(img0, imgGray);
                }
            }

            return 0;
        } else if (s == 'v')
        {}
        else if (s == 'c')
        {}
        else cout << "Incorrect parameters";
    }
    else if (s == 'k')
    {
        std::string imageName;
        std::cout << "Enter the name of the image" << std::endl;
        std::cin >> imageName;
        KmeansData::image = imread("pics/" + imageName, cv::IMREAD_COLOR);
        if (!KmeansData::image.data)
        {
            std::cout << "Invalid name" << std::endl;
            cv::destroyAllWindows();
            return EXIT_FAILURE;
        }
        cv::namedWindow("Kmeans", cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("Clusters", "Kmeans", &KmeansData::sliderValue, KmeansData::SLIDER_MAX, trackBarMovement);
        cv::waitKey();
        cv::destroyAllWindows();
        return EXIT_SUCCESS;
    }
}