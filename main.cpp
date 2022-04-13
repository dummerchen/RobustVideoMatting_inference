#include<bits\stdc++.h>
#include "rvm.h"
#include<opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<gflags\gflags.h>
#include<codecvt>

using namespace std;

//DEFINE_string(onnx_path, "./onnx/rvm_mobilenetv3_fp32.onnx", "where is onnx model path");
DEFINE_string(onnx_path, "./onnx/epoch.onnx", "where is onnx model path");
//DEFINE_string(test_path, "./TEST_01.mp4", "test path: image(png,jpg) or mp4");
DEFINE_string(test_path, "./1.png", "test path: image(png,jpg) or mp4");
DEFINE_string(output_path, "./processed_1.png", "ouput file path ,default at current dir");
DEFINE_int32(num_thread, 6, " threads nums, use num_thread to inference");
DEFINE_double(downsample_ratio,0.25, "downsample ratio,the smaller the more fps but lowerer resolution");
DEFINE_bool(rgb, false, "default output only mask");

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
        rvm.detect(img_bgr, content, FLAGS_downsample_ratio);
        
        if (!FLAGS_rgb)
            // 预测的前景pha
            cv::imwrite(FLAGS_output_path, content.pha_mat*255.);
        else
            // 合成图
            cv::imwrite(FLAGS_output_path,content.merge_mat);
    }
    else
    {
        image_or_video = "video";
        rvm.detect_video(
            FLAGS_test_path,
            FLAGS_output_path,
            FLAGS_downsample_ratio, 30
        );
        
    }
	return 0;
}