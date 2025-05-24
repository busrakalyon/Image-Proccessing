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


using namespace cv;
using namespace std;

// RGB to Intensity dönüşümü
unsigned char rgbToGray(int r, int g, int b) {
    return static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
}

// RGB'den griye çevirme fonksiyonu
Mat convertToGray(const Mat& img) {
    Mat grayImg(img.rows, img.cols, CV_8UC1); // Gri görüntü için matris oluştur

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
    // Kullanıcı bir tuşa basana kadar bekle
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
            // Açıyı 0-180 aralığına getir 
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
                suppressed.at<uchar>(i, j) = static_cast<uchar>(mag); // Orijinal magnitude değerini sakla
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

    int targetPixels = static_cast<int>(totalPixels * percentage);

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





// Daire tespiti 
vector<Vec3f> detectCircles(const Mat& binary, int minRadius, int maxRadius, int threshold) {
    vector<Vec3f> circles;
    int rows = binary.rows;
    int cols = binary.cols;
    int radiusRange = maxRadius - minRadius + 1;


    vector<Mat> houghSpace(radiusRange);
    for (int r = 0; r < radiusRange; r++) {
        houghSpace[r] = Mat::zeros(rows, cols, CV_32SC1);
    }

    // Kenar piksellerini dolaş ve oyları topla
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (binary.at<uchar>(y, x) == 255) {
                // Her bir kenar pikseli için olası tüm çember merkezlerini hesapla
                for (int r = 0; r < radiusRange; r++) {
                    int radius = r + minRadius;

                    // Açı artış miktarını yarıçapa göre ayarla (daha büyük yarıçaplar için daha az açı)
                    int angleStep = max(1, 360 / (4 * radius)); 

                    // Her bir açı için
                    for (int theta = 0; theta < 360; theta += angleStep) {
                        // Radyana çevir
                        double rad = theta * PI / 180.0;

                        // Potansiyel merkez koordinatlarını hesapla
                        int a = cvRound(x - radius * cos(rad));
                        int b = cvRound(y - radius * sin(rad));

                        // Görüntü sınırları içinde mi?
                        if (a >= 0 && a < cols && b >= 0 && b < rows) {
                            houghSpace[r].at<int>(b, a)++;
                        }
                    }
                }
            }
        }
    }

    Mat houghSpaceVis = Mat::zeros(rows, cols, CV_8UC1);
    for (int r = 0; r < radiusRange; r++) {
        Mat temp;
        normalize(houghSpace[r], temp, 0, 255, NORM_MINMAX);
        temp.convertTo(temp, CV_8UC1);
        // Her bir yarıçap için en yüksek değerleri al
        houghSpaceVis = max(houghSpaceVis, temp);
    }

    imshow("Hough Circle Transform", houghSpaceVis);
    waitKey(0);
    imwrite("resource/HoughCircleTransform.png", houghSpaceVis);


    // Geçici sonuçları sakla
    vector<Vec4f> tempCircles; // (x, y, radius, votes)


    // Eşik değerini geçen noktaları tespit et
    for (int r = 0; r < radiusRange; r++) {
        int radius = r + minRadius;
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int votes = houghSpace[r].at<int>(y, x);
                if (votes > threshold) {
                    // Yerel maksimum kontrolü - daha geniş alan (5x5)
                    bool isLocalMax = true;

                    int windowSize = 5; // Pencere boyutu
                    int halfWindow = windowSize / 2;

                    // Aynı yarıçap için daha büyük pencerede maksimum kontrolü
                    for (int ny = max(0, y - halfWindow); ny <= min(rows - 1, y + halfWindow) && isLocalMax; ny++) {
                        for (int nx = max(0, x - halfWindow); nx <= min(cols - 1, x + halfWindow) && isLocalMax; nx++) {
                            if ((ny != y || nx != x) && houghSpace[r].at<int>(ny, nx) > votes) {
                                isLocalMax = false;
                            }
                        }
                    }

                    // komşu yarıçaplar için de kontrol
                    if (isLocalMax) {
                        for (int nr = max(0, r - 2); nr <= min(radiusRange - 1, r + 2) && isLocalMax; nr++) {
                            if (nr != r) {
                                // Civardaki yarıçaplarda daha güçlü bir merkez var mı?
                                if (houghSpace[nr].at<int>(y, x) > votes) {
                                    isLocalMax = false;
                                }
                            }
                        }
                    }

                    // Eğer yerel maksimumsa, geçici listeye ekle
                    if (isLocalMax) {
                        tempCircles.push_back(Vec4f(x, y, radius, votes));
                    }
                }
            }
        }
    }

    // Oy sayısına göre sırala (en çok oydan en aza)
    sort(tempCircles.begin(), tempCircles.end(),
        [](const Vec4f& a, const Vec4f& b) { return a[3] > b[3]; });

    // Yakın daireleri birleştir
    for (size_t i = 0; i < tempCircles.size(); i++) {
        bool shouldAdd = true;

        // Bu merkez daha önce eklenen başka bir daireye çok yakın mı?
        for (size_t j = 0; j < circles.size(); j++) {
            float centerDist = sqrt(pow(tempCircles[i][0] - circles[j][0], 2) +
                pow(tempCircles[i][1] - circles[j][1], 2));
            float radiusDiff = abs(tempCircles[i][2] - circles[j][2]);

            // Eğer merkezler yeterince yakın ve yarıçaplar benzer ise
            if (centerDist < tempCircles[i][2] * 0.5 && radiusDiff < tempCircles[i][2] * 0.5) {
                shouldAdd = false;
                break;
            }
        }

        if (shouldAdd) {
            circles.push_back(Vec3f(tempCircles[i][0], tempCircles[i][1], tempCircles[i][2]));
        }
    }

    // En iyi N daire ile sınırla 
    const int maxCircles = 12;
    if (circles.size() > maxCircles) {
        circles.resize(maxCircles);
    }

    return circles;
}

// Hough dönüşümü ile tespit edilen daireleri çiz
void drawHoughCircles(Mat& image, const vector<Vec3f>& circles) {
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // Merkezi çiz
        circle(image, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        // Daireyi çiz
        circle(image, center, radius, Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("Detected Circles", image);
    waitKey(0);
    imwrite("resource/detected_circles.png", image);
}

int main() {
    string path = "resource/a.png";
    Mat img = imread(path, IMREAD_UNCHANGED);

    // Görüntü yüklendi mi kontrol et
    if (img.empty()) {
        cout << "Goruntu yuklenemedi!" << endl;
        return 0;
    }

    // Görüntüyü griye çevir
    Mat grayImg = convertToGray(img);
    // Kenar tespiti için önce blur uygula
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

    // Hough dönüşümü için parametreler - görüntüye göre ayarlanabilir
    int minRadius = 15;
    int maxRadius = 150;
    int minVotes = 80; 

    // Hough dönüşümü ile daire tespiti
   // Mat houghSpace = houghCircleTransform(binary, minRadius, maxRadius);

    // Daireleri tespit et
    vector<Vec3f> circles = detectCircles(binary, minRadius, maxRadius, minVotes);

    // Tespit edilen daire sayısını raporla
    cout << "Tespit edilen daire sayısı: " << circles.size() << endl;

    // Tespit edilen daireleri çiz
    Mat imgCopy = img.clone();
    drawHoughCircles(imgCopy, circles);

    return 0;
}