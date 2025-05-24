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



//void drawHistogram(const Mat& hist) {
//    int hist_w = 512; // Histogram genişliği
//    int hist_h = 400; // Histogram yüksekliği
//    int bin_w = cvRound((double)hist_w / 256); // Her bir kutunun genişliği
//    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0)); // Histogram görüntüsü
//    // Histogramı çiz
//    for (int i = 1; i < 256; i++) {
//        line(histImage,
//            Point(bin_w * (i - 1), hist_h - cvRound(hist.at<int>(0, i - 1))),
//            Point(bin_w * (i), hist_h - cvRound(hist.at<int>(0, i))),
//            Scalar(255, 0, 0), 2);
//    }
//    imshow("Histogram", histImage);
//    waitKey(0);
//}



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
                    distance = i * cos(k * PI / 180) + j * sin(k * PI / 180);
                    int rho = cvRound(distance);

                    if (rho >= 0 && rho < d) {
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

    // Piksel taraması
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

    // Eşik değerini geçen noktaları toplama
    for (int rho = 0; rho < accumulator.rows; rho++) {
        for (int theta = 0; theta < accumulator.cols; theta++) {
            if (accumulator.at<int>(rho, theta) >= threshold) {
                lines.push_back(Vec2f((rho - rho_bins) * rho_res, theta * theta_res));
            }
        }
    }
}


// Standart Hough dönüşümü ile çizgi çizme
void drawHoughLines(Mat& image, const Mat& binary, vector<Vec2f>& lines) {

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



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Çizgilerin kesişim noktalarını hesaplama fonksiyonu
Point2f lineIntersection(const Vec2f& line1, const Vec2f& line2) {
    // Hough çizgileri polar koordinat sisteminde (rho, theta) olarak temsil edilir
    float rho1 = line1[0], theta1 = line1[1];
    float rho2 = line2[0], theta2 = line2[1];

    // Kesişim noktasını hesaplamak için x*cos(theta) + y*sin(theta) = rho formülünü kullanırız
    double a1 = cos(theta1), b1 = sin(theta1);
    double a2 = cos(theta2), b2 = sin(theta2);

    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < 1e-6) { // Çizgiler paralel veya çakışık //1e-6 - bilimsel gösterimle 0.000001
        return Point2f(-1, -1);
    }

    double x = (b2 * rho1 - b1 * rho2) / det;
    double y = (a1 * rho2 - a2 * rho1) / det;

    return Point2f(x, y);
}

// Benzer çizgileri filtreleme fonksiyonu
vector<Vec2f> filterSimilarLines(const vector<Vec2f>& lines, float rhoThreshold, float thetaThreshold) {
    if (lines.empty()) return vector<Vec2f>();

    vector<Vec2f> filteredLines;
    filteredLines.push_back(lines[0]);

    for (size_t i = 1; i < lines.size(); i++) {
        bool isDuplicate = false;
        for (const auto& existingLine : filteredLines) {
            float rhoDiff = fabs(lines[i][0] - existingLine[0]);
            float thetaDiff = fabs(lines[i][1] - existingLine[1]);

            // Theta için sarmal farkı hesapla (0 ve PI arasında)
            if (thetaDiff > CV_PI / 2) {
                thetaDiff = CV_PI - thetaDiff;
            }

            if (rhoDiff < rhoThreshold && thetaDiff < thetaThreshold) {
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate) {
            filteredLines.push_back(lines[i]);
        }
    }

    return filteredLines;
}

// Yatay ve dikey çizgileri ayırma fonksiyonu
void separateHorizontalVerticalLines(const vector<Vec2f>& lines,
    vector<Vec2f>& horizontalLines,
    vector<Vec2f>& verticalLines,
    float thetaThreshold = 0.1) {

    horizontalLines.clear();
    verticalLines.clear();

    for (const auto& line : lines) {
        float theta = line[1];

        // Dikey çizgiler için theta yaklaşık 0 veya PI'dır
        if (theta < thetaThreshold || fabs(theta - CV_PI) < thetaThreshold) {
            verticalLines.push_back(line);
        }
        // Yatay çizgiler için theta yaklaşık PI/2 veya 3*PI/2'dir
        else if (fabs(theta - CV_PI / 2) < thetaThreshold || fabs(theta - 3 * CV_PI / 2) < thetaThreshold) {
            horizontalLines.push_back(line);
        }
    }

    // Çizgileri rho değerine göre sırala
    sort(horizontalLines.begin(), horizontalLines.end(), [](const Vec2f& a, const Vec2f& b) {
        return a[0] < b[0];
        });

    sort(verticalLines.begin(), verticalLines.end(), [](const Vec2f& a, const Vec2f& b) {
        return a[0] < b[0];
        });
}

// Kesişim noktalarını bulan fonksiyon
vector<Point2f> findIntersectionPoints(const vector<Vec2f>& horizontalLines,
    const vector<Vec2f>& verticalLines,
    const Size& imgSize) {
    vector<Point2f> intersections;

    // Boş vektör kontrolü
    if (horizontalLines.empty() || verticalLines.empty()) {
        return intersections;
    }

    for (const auto& hLine : horizontalLines) {
        for (const auto& vLine : verticalLines) {
            Point2f pt = lineIntersection(hLine, vLine);

            // Geçerli kesişim noktası ve görüntü sınırları içinde olmalı
            if (pt.x >= 0 && pt.y >= 0 && pt.x < imgSize.width && pt.y < imgSize.height) {
                intersections.push_back(pt);
            }
        }
    }

    return intersections;
}

// Kesişim noktalarını kare köşelerine göre düzenleyen fonksiyon
vector<vector<Point2f>> organizeIntersectionsToGrid(const vector<Point2f>& intersections,
    int expectedRows, int expectedCols) {
    // Noktaları y koordinatına göre kümelemek için
    const double yTolerance = 10.0; // pixel

    // Boş vektör kontrolü
    if (intersections.empty()) {
        return vector<vector<Point2f>>();
    }

    // Önce y koordinatına göre grupla (satırlar)
    map<int, vector<Point2f>> rowClusters;
    for (const auto& pt : intersections) {
        bool addedToCluster = false;
        for (auto& cluster : rowClusters) {
            if (fabs(pt.y - cluster.first) < yTolerance) {
                cluster.second.push_back(pt);
                addedToCluster = true;
                break;
            }
        }

        if (!addedToCluster) {
            rowClusters[static_cast<int>(pt.y)] = { pt };
        }
    }

    // Satırları y koordinatına göre sırala
    vector<vector<Point2f>> rows;
    for (auto& cluster : rowClusters) {
        // Her satırı x koordinatına göre sırala
        sort(cluster.second.begin(), cluster.second.end(), [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
            });
        rows.push_back(cluster.second);
    }

    // Satırları y koordinatına göre sırala
    sort(rows.begin(), rows.end(), [](const vector<Point2f>& a, const vector<Point2f>& b) {
        // Boş vektör kontrolü
        if (a.empty() || b.empty()) return false;
        return a[0].y < b[0].y;
        });

    // Beklenen satır ve sütun sayısını kontrol et
    if (static_cast<int>(rows.size()) != expectedRows ||
        any_of(rows.begin(), rows.end(), [expectedCols](const vector<Point2f>& row) {
            return static_cast<int>(row.size()) != expectedCols;
            })) {
        cerr << "Uyarı: Beklenen ızgara boyutu (" << expectedRows << "x" << expectedCols
            << ") ile bulunan ızgara boyutu (" << rows.size() << "x";

        if (!rows.empty()) {
            cerr << rows[0].size();
        }
        else {
            cerr << "0";
        }

        cerr << ") eşleşmiyor!" << endl;
    }

    return rows;
}

// Kalibrasyon matrisini hesaplayan fonksiyon (C = A*W)
// C: Görüntü koordinatları, W: Gerçek dünya koordinatları, A: Kalibrasyon matrisi katsayıları
Mat calculateCalibrationMatrix(const vector<Point2f>& imagePoints, const vector<Point2f>& worldPoints) {
    // En az 4 nokta gerekir
    if (imagePoints.size() < 4 || worldPoints.size() < 4 || imagePoints.size() != worldPoints.size()) {
        cerr << "Kalibrasyon için yetersiz eşleşme noktası!" << endl;
        return Mat();
    }

    // A*X = B formunda bir denklemi çözmek için
    // Burada X bizim kalibrasyon katsayılarımız olacak
    Mat A = Mat::zeros(2 * imagePoints.size(), 8, CV_64F); //2n+8 boyutlu matris
    Mat B = Mat::zeros(2 * imagePoints.size(), 1, CV_64F); //2n+1 boyutlu matris

    for (size_t i = 0; i < imagePoints.size(); i++) {
        double u = imagePoints[i].x;
        double v = imagePoints[i].y;
        double x = worldPoints[i].x;
        double y = worldPoints[i].y;

        // İlk denklem: u = a1*x + a2*y + a3*x*y + a4
        A.at<double>(i * 2, 0) = x;
        A.at<double>(i * 2, 1) = y;
        A.at<double>(i * 2, 2) = x * y;
        A.at<double>(i * 2, 3) = 1;
        B.at<double>(i * 2, 0) = u;

        // İkinci denklem: v = a5*x + a6*y + a7*x*y + a8
        A.at<double>(i * 2 + 1, 4) = x;
        A.at<double>(i * 2 + 1, 5) = y;
        A.at<double>(i * 2 + 1, 6) = x * y;
        A.at<double>(i * 2 + 1, 7) = 1;
        B.at<double>(i * 2 + 1, 0) = v;
    }

    // En küçük kareler yöntemiyle çözüm
    Mat X;
    solve(A, B, X, DECOMP_SVD);

    return X;
}

// Gerçek dünya koordinatlarını hesaplayan fonksiyon
Point2f calculateRealWorldCoordinates(const Point2f& imagePoint, const Mat& calibrationMatrix) {
    if (calibrationMatrix.empty()) {
        cerr << "Hata: Kalibrasyon matrisi boş!" << endl;
        return Point2f(-1, -1);
    }

    double u = imagePoint.x;
    double v = imagePoint.y;

    // a1 ile a8 değerlerini al
    double a1 = calibrationMatrix.at<double>(0, 0);
    double a2 = calibrationMatrix.at<double>(1, 0);
    double a3 = calibrationMatrix.at<double>(2, 0);
    double a4 = calibrationMatrix.at<double>(3, 0);
    double a5 = calibrationMatrix.at<double>(4, 0);
    double a6 = calibrationMatrix.at<double>(5, 0);
    double a7 = calibrationMatrix.at<double>(6, 0);
    double a8 = calibrationMatrix.at<double>(7, 0);

    // Başlangıç tahminleri (sıfıra bölünme kontrolü)
    double x0 = (fabs(a1) > 1e-6) ? u / a1 : 0;
    double y0 = (fabs(a6) > 1e-6) ? v / a6 : 0;

    // İteratif çözüm için Newton-Raphson yöntemi
    double x = x0, y = y0;
    for (int i = 0; i < 10; i++) {  // 10 iterasyon genellikle yeterli
        // Fonksiyon değerleri
        double f1 = a1 * x + a2 * y + a3 * x * y + a4 - u;
        double f2 = a5 * x + a6 * y + a7 * x * y + a8 - v;

        // Jacobian matrisi
        double j11 = a1 + a3 * y;
        double j12 = a2 + a3 * x;
        double j21 = a5 + a7 * y;
        double j22 = a6 + a7 * x;

        // Determinant
        double det = j11 * j22 - j12 * j21;
        if (fabs(det) < 1e-10) break;

        // Delta x ve y hesapla
        double dx = (j22 * f1 - j12 * f2) / det;
        double dy = (j11 * f2 - j21 * f1) / det;

        // Güncelle
        x -= dx;
        y -= dy;

        // Yakınsama kontrolü
        if (fabs(dx) < 1e-6 && fabs(dy) < 1e-6) break;
    }

    return Point2f(x, y);
}

// Seçilen alanın mm cinsinden ölçüsünü hesaplayan fonksiyon
void calculateAreaDimensions(const vector<Point2f>& imagePoints, const Mat& calibrationMatrix) {
    if (imagePoints.size() < 3) {
        cerr << "Alan ölçümü için en az 3 nokta gerekiyor!" << endl;
        return;
    }

    if (calibrationMatrix.empty()) {
        cerr << "Hata: Kalibrasyon matrisi boş!" << endl;
        return;
    }

    // Her nokta için gerçek dünya koordinatlarını hesapla
    vector<Point2f> worldPoints;
    for (const auto& pt : imagePoints) {
        Point2f worldPt = calculateRealWorldCoordinates(pt, calibrationMatrix);
        if (worldPt.x >= 0 && worldPt.y >= 0) { // Geçerli nokta kontrolü
            worldPoints.push_back(worldPt);
        }
    }

    if (worldPoints.size() < 3) {
        cerr << "Yeterli geçerli nokta bulunamadı!" << endl;
        return;
    }

    int n = worldPoints.size();

    // En ve boy hesapla (basit dikdörtgen yaklaşımı)
    double minX = worldPoints[0].x, maxX = worldPoints[0].x;
    double minY = worldPoints[0].y, maxY = worldPoints[0].y;

    for (const auto& pt : worldPoints) {
        minX = fmin(minX, pt.x);
        maxX = fmax(maxX, pt.x);
        minY = fmin(minY, pt.y);
        maxY = fmax(maxY, pt.y);
    }

    double width = maxX - minX;
    double height = maxY - minY;

    cout << "\nAlan ölçüleri:" << endl;
    cout << "Genişlik: " << width << " mm" << endl;
    cout << "Yükseklik: " << height << " mm" << endl;
}

// Fare ile seçim için callback fonksiyonu
struct MouseData {
    vector<Point> selectedPoints;
    Mat image;
    Mat originalImage; // Orjinal görüntüyü saklama
    bool selectionComplete = false;
};

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    MouseData* data = static_cast<MouseData*>(userdata);

    if (!data || data->image.empty()) {
        return;
    }

    if (event == EVENT_LBUTTONDOWN) {
        // Nokta ekle
        data->selectedPoints.push_back(Point(x, y));

        // Noktayı göster
        circle(data->image, Point(x, y), 5, Scalar(0, 0, 255), -1);

        // Son iki nokta arasında çizgi çiz
        if (data->selectedPoints.size() > 1) {
            int last = data->selectedPoints.size() - 1;
            line(data->image, data->selectedPoints[last - 1], data->selectedPoints[last],
                Scalar(0, 255, 0), 2);
        }

        imshow("Alan Seçimi", data->image);
    }
    else if (event == EVENT_RBUTTONDOWN) {
        // Seçimi tamamla ve çokgeni kapat
        if (data->selectedPoints.size() > 2) {
            // İlk ve son noktayı birleştir
            line(data->image, data->selectedPoints.back(), data->selectedPoints.front(),
                Scalar(0, 255, 0), 2);

            imshow("Alan Seçimi", data->image);
            data->selectionComplete = true;
        }
        else {
            cout << "En az 3 nokta gerekiyor! Şu anda: " << data->selectedPoints.size() << endl;
        }
    }
    else if (event == EVENT_MBUTTONDOWN) {
        // Seçimi sıfırla (orta tuş)
        data->selectedPoints.clear();
        data->image = data->originalImage.clone();
        data->selectionComplete = false;
        imshow("Alan Seçimi", data->image);
        cout << "Seçim sıfırlandı. Yeniden başlayın." << endl;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() {

    std::locale::global(std::locale("")); //türkçe karakter desteği için

    string path = "resource/a2.jpg";
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
    vector<Vec2f> lines;
    drawHoughLines(imgCopy, binary, lines);

    vector<Vec2f> filteredLines = filterSimilarLines(lines, 30, 0.1);

    // Yatay ve dikey çizgileri ayır
    vector<Vec2f> horizontalLines, verticalLines;
    separateHorizontalVerticalLines(filteredLines, horizontalLines, verticalLines);

    cout << "Yatay çizgi sayısı: " << horizontalLines.size() << endl;
    cout << "Dikey çizgi sayısı: " << verticalLines.size() << endl;


    // Kesişim noktalarını bul
    vector<Point2f> intersections = findIntersectionPoints(horizontalLines, verticalLines, img.size());

    cout << "Tespit edilen kesişim noktası sayısı: " << intersections.size() << endl;

    // Kesişim noktalarını göster
    Mat intersectionImage = img.clone();
    for (const auto& pt : intersections) {
        circle(intersectionImage, pt, 3, Scalar(0, 255, 0), -1);
    }

    imshow("Kesişim Noktaları", intersectionImage);

    // Dama tahtası boyutlarını belirle (satır ve sütun sayısı)
    int expectedRows = 7;  // 8x8 dama tahtası için 7x7 iç kesişim noktası
    int expectedCols = 7;

    // Kesişim noktalarını ızgara şeklinde düzenle
    vector<vector<Point2f>> intersectionGrid = organizeIntersectionsToGrid(
        intersections, expectedRows, expectedCols);

    // Gerçek dünya koordinatları (mm cinsinden)
    vector<Point2f> imagePoints;
    vector<Point2f> worldPoints;

    float squareSizeMM = 27.5f;  // Dama tahtası kare boyutu 2,75(mm)

    // Kalibrasyon için köşe noktalarını topla
    for (int i = 0; i < min(expectedRows, static_cast<int>(intersectionGrid.size())); i++) {
        for (int j = 0; j < min(expectedCols, static_cast<int>(intersectionGrid[i].size())); j++) {
            imagePoints.push_back(intersectionGrid[i][j]);
            worldPoints.push_back(Point2f(j * squareSizeMM, i * squareSizeMM));
        }
    }


    // Kalibrasyon matrisini hesapla
    Mat calibrationMatrix = calculateCalibrationMatrix(imagePoints, worldPoints);

    if (calibrationMatrix.empty()) {
        cerr << "Kalibrasyon matrisi hesaplanamadı!" << endl;
        return -1;
    }

    cout << "Kalibrasyon matrisi katsayıları:" << endl;
    for (int i = 0; i < calibrationMatrix.rows; i++) {
        cout << "a" << (i + 1) << ": " << calibrationMatrix.at<double>(i, 0) << endl;
    }

    // Kullanıcıdan alan seçmesini iste
    MouseData mouseData;
    mouseData.image = img.clone();
    namedWindow("Alan Seçimi", WINDOW_AUTOSIZE);
    setMouseCallback("Alan Seçimi", mouseCallback, &mouseData);

    cout << "\nAlan seçimi için talimatlar:" << endl;
    cout << "- Sol tıklama: Nokta ekle" << endl;
    cout << "- Sağ tıklama: Seçimi tamamla" << endl;

    imshow("Alan Seçimi", mouseData.image);

    // Kullanıcı seçimi tamamlayana kadar bekle
    while (!mouseData.selectionComplete) {
        int key = waitKey(100);
        if (key == 27) break;  // ESC tuşu ile çık
    }

    // Seçilen noktaları gerçek dünya koordinatlarına dönüştür ve alan ölçülerini hesapla
    if (mouseData.selectionComplete && !mouseData.selectedPoints.empty()) {
        vector<Point2f> selectedPointsFloat;
        for (const auto& pt : mouseData.selectedPoints) {
            selectedPointsFloat.push_back(Point2f(pt.x, pt.y));
        }

        calculateAreaDimensions(selectedPointsFloat, calibrationMatrix);
    }

    waitKey(0);

    return 0;
}