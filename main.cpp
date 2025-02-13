//yolo目标识别c++推理
//onnxruntime model_size = 1x5x8400

#include<iostream>
#include<vector>
#include<string>
#include<opencv.hpp>
#include<opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

void draw_center(int x, int y, Mat& image) {
    circle(image, Point(x, y), 5, Scalar(255, 0, 0), 6);
}

void draw_box(const Mat &img, vector<vector<float> > box_info, const float thinkness, const float text_size) {
    Point p1, p2;
    for (int i = 0; i < box_info.size(); i++) {
        p1 = Point(box_info[i][0], box_info[i][1]);
        p2 = Point(box_info[i][2], box_info[i][3]);
        float conf = box_info[i][4];
        string conf_text = to_string(conf);
        rectangle(img, p1, p2, Scalar(0, 255, 0), thinkness, 8);
        putText(img, conf_text, p1, cv::FONT_HERSHEY_SIMPLEX, text_size, Scalar(0, 255, 0), thinkness, 8);
    }
    
}

vector<vector<float> > process_result(const cv::Mat& outs, const float &confidence_threshold = 0.5) {
    vector<vector<float> > result_info;
    int batch_size = outs.size[0];
    int num_attributes = outs.size[1];
    int num_anchors = outs.size[2];

    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of attributes per anchor: " << num_attributes << std::endl;
    std::cout << "Number of anchors: " << num_anchors << std::endl;

    float* pdata = (float*)outs.data;

    vector<int> result_idx;
    vector<float> result_conf;
    //cout << pdata[16798] << endl;
    for (int i = 6720 * num_attributes; i <= num_anchors * num_attributes; i++) {
        if (pdata[i] > confidence_threshold) {
            result_idx.push_back(i - 6720 * num_attributes);
            result_conf.push_back(pdata[i]);
            cout << "conf:" << pdata[i] << endl;
            cout << "IDX:" << i - 6720 * num_attributes << endl;
        }
    }

    for (int i = 0; i < result_idx.size(); i++) {
        vector<float> row;
        int idx = result_idx[i];
        float x = pdata[idx];
        float y = pdata[8400 + idx];
        float w = pdata[16800 + idx];
        float h = pdata[25200 + idx];
        row.push_back(x - w / 2);
        row.push_back(y - h / 2);
        row.push_back(x + w / 2);
        row.push_back(y + h / 2);
        row.push_back(result_conf[i]);
        result_info.push_back(row);
        printf("idx:%d, x:%.2f, y:%.2f, w:%.2f, h:%.2f\n", result_idx[i], x, y, w, h);
    }
    return result_info;

    //33600---6720
    //8400
}

float get_iou(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
    float area_box_a = (p2.x - p1.x) * (p2.y - p1.y);
    float area_box_b = (p4.x - p3.x) * (p4.y - p3.y);

    float x1 = max(p1.x, p3.x);
    float y1 = max(p1.y, p3.y);
    float x2 = min(p2.x, p4.x);
    float y2 = min(p2.y, p4.y);
    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float area_intersection = width * height;
    float area_union = area_box_a + area_box_b - area_intersection;
    if (area_union == 0) return 0;

    float iou = area_intersection / area_union;
    return iou;
}

vector<vector<float> > nms(const vector<vector<float> >& result_info, float iou_threshold = 0.5) {
    vector<vector<float> > nms_box_info;

    if (result_info.empty()) return nms_box_info;

    vector<pair<int, float> > confidences;
    for (int i = 0; i < result_info.size(); ++i) {
        confidences.emplace_back(i, result_info[i][4]);
    }
    sort(confidences.begin(), confidences.end(), [](const pair<int, float>& a, const pair<int, float>& b) {
        return a.second > b.second;
        });

    vector<bool> suppressed(result_info.size(), false);


    for (int i = 0; i < confidences.size(); ++i) {
        int index = confidences[i].first;
        if (suppressed[index]) continue;

        nms_box_info.push_back(result_info[index]);

        for (int j = i + 1; j < confidences.size(); ++j) {
            int other_index = confidences[j].first;
            if (suppressed[other_index]) continue;

            Point p1(result_info[index][0], result_info[index][1]);
            Point p2(result_info[index][2], result_info[index][3]);
            Point p3(result_info[other_index][0], result_info[other_index][1]);
            Point p4(result_info[other_index][2], result_info[other_index][3]);

            float iou = get_iou(p1, p2, p3, p4);

            if (iou > iou_threshold) {
                suppressed[other_index] = true;
            }
        }
    }

    return nms_box_info;
}

int main() {

    cv::dnn::Net net = cv::dnn::readNetFromONNX("D:/yyj05/Documents/力创视觉组/OpenCVProject/vision_buffDetect/model/best.onnx");
    Mat src = imread("D:/yyj05/Documents/力创视觉组/OpenCVProject/vision_buffDetect/img/images/30.jpg");
    Mat src_resize;

    resize(src, src_resize, Size(640, 640));
    Mat blob = cv::dnn::blobFromImage(src_resize, 1.0 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);

    net.setInput(blob);
    vector<Mat> outs;
    vector<string> outNames = { "output0" };
    net.forward(outs, outNames);

    float confThreshold = 0.8;
    vector<vector<float> > result_info = process_result(outs[0], confThreshold);
    vector<vector<float> > nms_box = nms(result_info);
    draw_box(src_resize, nms_box, 1.7, 0.5);


    imshow("src", src_resize);

    waitKey(0);
    return 0;
}
