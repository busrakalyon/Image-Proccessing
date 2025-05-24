#include <opencv2/imgcodecs.hpp> // Görüntü dosyalarını yüklemek ve kaydetmek için
#include <opencv2/highgui.hpp>   // Görüntüleri ekranda göstermek için
#include <opencv2/imgproc.hpp>   // Görüntü işleme fonksiyonları
#include <opencv2/objdetect.hpp> // Nesne algılama işlemleri için
#include <opencv2/opencv.hpp>
#include <iostream>              // Konsola çıktı vermek için
#include <cmath>
#include <vector>
#include <climits>


constexpr double PI = 3.14159265358979323846;
//int* gradientDirection;

using namespace cv;
using namespace std;

// RGB to Intensity dönüşümü
unsigned char rgbToGray(int r, int g, int b) {
    return static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
}

// RGB'den griye çevirme fonksiyonu
Mat convertToGray(const Mat& img) {
    Mat grayImg(img.rows, img.cols, CV_8UC1); // Gri görüntü için matris 

    unsigned char* grayData = grayImg.data;
    unsigned char* imgData = img.data;
    int channels = img.channels();
    int width = img.cols;
    int height = img.rows;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * channels; // Pikselin başlangıç indexi
            int grayIndex = i * width + j;

            int blue = imgData[index];
            int green = imgData[index + 1];
            int red = imgData[index + 2];

            // RGB'den gri tona çevir ve yeni matrise ata
            grayData[grayIndex] = rgbToGray(red, green, blue);
        }
    }

    // Görüntüyü göster
    imshow("Gray Image", grayImg);
    waitKey(0);
    //kaydet
    imwrite("resource/test_gray.png", grayImg);

    return grayImg;
}




Mat GradientImage(const Mat& grayImg, Mat& gradientDirection) {
    // Gradyanları hesapla
    Mat gradX, gradY;

    int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    };

    int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Çıkış matrislerini oluştur
    gradX = Mat::zeros(grayImg.rows - 2, grayImg.cols - 2, CV_32F);
    gradY = Mat::zeros(grayImg.rows - 2, grayImg.cols - 2, CV_32F);

    // Görüntünün iç piksellerini dolaş
    for (int r = 0; r < gradX.rows; r++) {
        for (int c = 0; c < gradX.cols; c++) {
            float gx = 0, gy = 0;

            // 3x3 çekirdeği uygula
            for (int m = 0; m < 3; m++) {
                for (int n = 0; n < 3; n++) {
                    uchar pixel = grayImg.at<uchar>(r + m, c + n);
                    gx += pixel * sobelX[m][n];
                    gy += pixel * sobelY[m][n];
                }
            }

            gradX.at<float>(r, c) = gx;
            gradY.at<float>(r, c) = gy;
        }
    }


    // Gradyan büyüklüğünü ve yönünü hesapla
    Mat gradientMagnitude;

    gradientMagnitude = Mat(gradX.size(), CV_32F);
    gradientDirection = Mat(gradX.size(), CV_32F);

    for (int r = 0; r < gradX.rows; r++) {
        for (int c = 0; c < gradX.cols; c++) {
            float gx = gradX.at<float>(r, c);
            float gy = gradY.at<float>(r, c);

            // Büyüklüğü hesapla
            gradientMagnitude.at<float>(r, c) = sqrt(gx * gx + gy * gy);

            // Açıyı hesapla
            float theta = atan2(gy, gx) * (180.0 / PI); 
            // Açıyı 0-180 aralığına getir (çizgilerin yönü değil doğrultusu önemli)
            if (theta < 0) theta += 180;

            gradientDirection.at<float>(r, c) = theta;

        }
    }
    // Görselleştirme için normalize et
    Mat gradMagVis, gradDirVis;
    normalize(gradientMagnitude, gradMagVis, 0, 255, NORM_MINMAX);
    gradMagVis.convertTo(gradMagVis, CV_8U);

    normalize(gradientDirection, gradDirVis, 0, 255, NORM_MINMAX);
    gradDirVis.convertTo(gradDirVis, CV_8U);

    // Sonuçları görüntüle
    imshow("Gradyan Büyüklüğü", gradMagVis);
    imshow("Gradyan Yönü", gradDirVis);

    // Görüntüyü kaydet
    imwrite("resource/gradient_magnitude.jpg", gradMagVis);
    imwrite("resource/gradient_direction.jpg", gradDirVis);

    waitKey(0); // Kullanıcının tuşa basmasını bekle

    return gradientMagnitude;
}




Mat nonMaximumSuppression(const Mat& magnitude, const Mat& angle) {
    Mat suppressed = Mat::zeros(magnitude.size(), CV_8UC1);

    for (int i = 1; i < magnitude.rows - 1; ++i) {
        for (int j = 1; j < magnitude.cols - 1; ++j) {
            float angleDeg = angle.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);

            // Normalize angle to [0,180)
            if (angleDeg < 0) angleDeg += 180;
            else if (angleDeg >= 180) angleDeg -= 180;

            float neighbor1 = 0, neighbor2 = 0;

            if ((angleDeg >= 0 && angleDeg < 22.5) || (angleDeg >= 157.5 && angleDeg < 180)) {
                // 0 derece (yatay)
                neighbor1 = magnitude.at<float>(i, j - 1);
                neighbor2 = magnitude.at<float>(i, j + 1);
            }
            else if (angleDeg >= 22.5 && angleDeg < 67.5) {
                // 45 derece (çapraz)
                neighbor1 = magnitude.at<float>(i - 1, j + 1);
                neighbor2 = magnitude.at<float>(i + 1, j - 1);
            }
            else if (angleDeg >= 67.5 && angleDeg < 112.5) {
                // 90 derece (dikey)
                neighbor1 = magnitude.at<float>(i - 1, j);
                neighbor2 = magnitude.at<float>(i + 1, j);
            }
            else if (angleDeg >= 112.5 && angleDeg < 157.5) {
                // 135 derece (çapraz)
                neighbor1 = magnitude.at<float>(i - 1, j - 1);
                neighbor2 = magnitude.at<float>(i + 1, j + 1);
            }

            if (mag >= neighbor1 && mag >= neighbor2)
                suppressed.at<uchar>(i, j) = static_cast<uchar>(mag); // mag, iki komşudan büyük veya eşitse, bu noktayı kenar olarak sakla, değilse bu noktayı sıfırla
        }
    }
    Mat suppressedNorm;
    normalize(suppressed, suppressedNorm, 0, 255, NORM_MINMAX);
    suppressedNorm.convertTo(suppressedNorm, CV_8U);

    imshow("Non-Maximum Suppressed", suppressedNorm);
    waitKey(0);
    imwrite("resource/suppressed.jpg", suppressedNorm);
    return suppressed;
}




Mat Histogram(const Mat& suppressedNorm) {
    Mat hist = Mat::zeros(1, 256, CV_32SC1); // 0-255 arası değerler için 256 kutu

    for (int i = 0; i < suppressedNorm.rows; i++) {
        for (int j = 0; j < suppressedNorm.cols; j++) {
            int pixel_value = suppressedNorm.at<uchar>(i, j);
            hist.at<int>(0, pixel_value)++;
        }
    }

    return hist;
}




int CalculateThresholdFromHistogram(const Mat& hist, float percentage) {
    int totalPixels = 0;
    for (int i = 0; i < 256; i++) {
        totalPixels += hist.at<int>(0, i);
    }

    int targetPixels = static_cast<int>(totalPixels * percentage); //percenage en parlak pixel %desi

    int cumulative = 0;
    for (int i = 255; i >= 0; i--) { // En parlaklardan başla
        cumulative += hist.at<int>(0, i);
        if (cumulative >= targetPixels) {
            return i; 
        }
    }

    return 0; 
}


Mat ApplyThresholdWithHysteresis(const Mat& suppressedNorm, int lowThreshold, int highThreshold) {
    Mat binary = Mat::zeros(suppressedNorm.size(), CV_8UC1);

    for (int i = 0; i < suppressedNorm.rows; i++) {
        for (int j = 0; j < suppressedNorm.cols; j++) {
            uchar pixel = suppressedNorm.at<uchar>(i, j);

            if (pixel >= highThreshold) {
                binary.at<uchar>(i, j) = 255; // Güçlü kenar
            }
            else if (pixel >= lowThreshold) {
                binary.at<uchar>(i, j) = 100; // Zayıf kenar 
            }
            else {
                binary.at<uchar>(i, j) = 0; // Arka plan
            }
        }
    }

    imshow("Threshold", binary);
    waitKey(0);
    imwrite("resource/Threshold.png", binary);

    return binary;
}


void edgesWithHysteresis(Mat& binary) {
    Mat output = binary.clone();
    int rows = binary.rows;
    int cols = binary.cols;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (binary.at<uchar>(i, j) == 100) { // Zayıf kenar
                // 8 komşuya bak
                if (
                    binary.at<uchar>(i - 1, j - 1) == 255 || binary.at<uchar>(i - 1, j) == 255 || binary.at<uchar>(i - 1, j + 1) == 255 ||
                    binary.at<uchar>(i, j - 1) == 255 || binary.at<uchar>(i, j + 1) == 255 ||
                    binary.at<uchar>(i + 1, j - 1) == 255 || binary.at<uchar>(i + 1, j) == 255 || binary.at<uchar>(i + 1, j + 1) == 255
                    ) {
                    output.at<uchar>(i, j) = 255; // Güçlü kenarla bağlantılıysa kenar olarak kabul et
                }
                else {
                    output.at<uchar>(i, j) = 0; // Bağlantısızsa arka plana at
                }
            }
        }
    }

    binary = output; // Güncellenmiş matrisi geri ata

    imshow("Hysteresis Result", binary);
    waitKey(0);
    imwrite("resource/hysteresis_edges.png", binary);
}



Mat houghTransform(const Mat& binaryImage) {
    int rows = binaryImage.rows;
    int cols = binaryImage.cols;
    // Uzaklık max köşegen kadar değer alabilir
    int d = (int)sqrt(rows * rows + cols * cols);
    Mat houghSpace = Mat::zeros(d, 180, CV_32SC1);

    // PI sabitini tanımlayalım
    const double PI = CV_PI;
    double distance = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (binaryImage.at<uchar>(i, j) == 255) {
                for (int k = 0; k < 180; k++) {
                    distance = i * cos(k * PI / 180) + j * sin(k * PI / 180); //ρ=x⋅cos(θ)+y⋅sin(θ)
                    int rho = cvRound(distance);

					if (rho >= 0 && rho < d) { //hesaplanan rho geçerli mi 0<rho<d
                        houghSpace.at<int>(rho, k) += 1;
                    }
                }
            }
        }
    }

    Mat houghSpaceVis;
    normalize(houghSpace, houghSpaceVis, 0, 255, NORM_MINMAX);
    houghSpaceVis.convertTo(houghSpaceVis, CV_8U);

    imshow("Hough Transform", houghSpaceVis);
    waitKey(0);

    imwrite("resource/HoughTransform.png", houghSpaceVis);
    return houghSpace;
}




void manualHoughLines(const Mat& binary, vector<Vec2f>& lines, int rho_res, double theta_res, int threshold) {
    int rows = binary.rows, cols = binary.cols;
    int max_rho = sqrt(rows * rows + cols * cols);
    int rho_bins = max_rho / rho_res;
    int theta_bins = CV_PI / theta_res;

    Mat accumulator = Mat::zeros(rho_bins * 2, theta_bins, CV_32S);

    // Piksel taraması, her beyaz pixel için potansiyel çizgilere oy verme
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (binary.at<uchar>(y, x) > 0) {
                for (int theta_idx = 0; theta_idx < theta_bins; theta_idx++) {
                    double theta = theta_idx * theta_res;
                    int rho = cvRound(x * cos(theta) + y * sin(theta)) + rho_bins;
                    accumulator.at<int>(rho, theta_idx)++;
                }
            }
        }
    }

    // Eşik değerini geçen potansiyel çizgi noktalarını toplama
    for (int rho = 0; rho < accumulator.rows; rho++) {
        for (int theta = 0; theta < accumulator.cols; theta++) {
            if (accumulator.at<int>(rho, theta) >= threshold) {
                lines.push_back(Vec2f((rho - rho_bins) * rho_res, theta * theta_res));
            }
        }
    }
}


// Standart Hough dönüşümü ile çizgi çizme
void drawHoughLines(Mat& image, const Mat& binary) {
    vector<Vec2f> lines;
    manualHoughLines(binary, lines, 1, CV_PI / 180, 110);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(image, pt1, pt2, Scalar(0, 0, 255), 2);
    }


    imshow("HoughLines", image);
    waitKey(0);
    imwrite("resource/houghlines.png", image);
}




int main() {
    string path = "resource/a2.png";
    Mat img = imread(path, IMREAD_UNCHANGED);

    // Görüntü yüklendi mi kontrol et
    if (img.empty()) {
        cout << "Goruntu yuklenemedi!" << endl;
        return 0;
    }

    // Görüntüyü griye çevir
    Mat grayImg = convertToGray(img);
    // Kenar tespiti
    GaussianBlur(grayImg, grayImg, Size(5, 5), 1.5);

    Mat gradientDirection;
    Mat gradientMagnitude = GradientImage(grayImg, gradientDirection);




    Mat NMS = nonMaximumSuppression(gradientMagnitude, gradientDirection);


    Mat hist = Histogram(NMS);
    //drawHistogram(hist);

    int threshold = CalculateThresholdFromHistogram(hist, 0.02);
    cout << "Hesaplanan Threshold = " << threshold << endl;

    int lowThreshold = 30;
    int highThreshold = 70;

    Mat binary = ApplyThresholdWithHysteresis(NMS, lowThreshold, highThreshold);


    edgesWithHysteresis(binary);



    Mat houghSpace = houghTransform(binary);




    //HoughLines fonksiyonu ile çizgi tespiti
    Mat imgCopy = img.clone();
    drawHoughLines(imgCopy, binary);




    return 0;
}