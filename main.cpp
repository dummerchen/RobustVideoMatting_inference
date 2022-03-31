#include<bits\stdc++.h>
#include "rvm.h"
#include<opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<gflags\gflags.h>
#include<codecvt>
using namespace std;

DEFINE_string(onnx_path, "./onnx/rvm_mobilenetv3_fp32.onnx", "model path");
//DEFINE_string(onnx_path, "./onnx/modnet.onnx", "model path");
//DEFINE_string(test_path, "E:\\py_exercise\\service_project/datasets/images/4.png", "test path");

DEFINE_string(test_path, "../datasets/test/TEST_18.mp4", "test path");
//DEFINE_string(output_path,"")
DEFINE_int32(num_thread, 6, "threads nums");

vector <string> split_name(string path)
{
    size_t pos = path.find_last_of('/');
    string name;
    if (pos != NULL)
    {
        name = path.substr(pos + 1);
    }
    else
    {
        size_t pos = path.find_last_of('\\');
        name = path.substr(pos + 1);
    }
    pos = name.find_last_of('.');
    string suffix = name.substr(pos + 1);
    return vector<string>{name, suffix};
}

int main(int argc,char ** argv)
{
    google::SetUsageMessage("Please attention  : string cmd line args without quota");
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    wstring_convert < codecvt_utf8_utf16<wchar_t> > converter;
    wstring path = converter.from_bytes(FLAGS_onnx_path);
    RobustVideoMatting rvm(path, FLAGS_num_thread); // 16 threads
    
    string image_or_video, suffix, name;
    vector<string> test_info = split_name(FLAGS_test_path);
    name = test_info[0];
    suffix = test_info[1];
    if ((suffix == "jpg") || (suffix == "png"))
    {
        MattingContent content;
        // 1. image matting.
        image_or_video = "image";
        cv::Mat img_bgr = cv::imread(FLAGS_test_path);
        clock_t start_time = clock();
        rvm.detect(img_bgr, content, 0.25f);
        clock_t end_time = clock();
        std::cout << (end_time - start_time) / 1000.0 <<" s/frame" << endl;
        // 预测的前景pha
         cv::imwrite("mask_" + test_info[0], content.pha_mat * 255.);
        // 合成图
        cv::imwrite("merge_" + test_info[0],content.merge_mat);
    }
    else
    {
        image_or_video = "video";
        rvm.detect_video(
            FLAGS_test_path,
            test_info[0],
            0.25, 30
        );
    }
    system("pause");
	return 0;
}