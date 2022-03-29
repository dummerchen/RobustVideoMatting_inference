#include<onnxruntime_cxx_api.h>
#include<bits/stdc++.h>
#include<opencv.hpp>
using namespace std;

typedef struct MattingContentType
{
    cv::Mat fgr_mat; // fore ground mat 3 channel (R,G,B) 0.~1. or 0~255
    cv::Mat pha_mat; // alpha(matte) 0.~1.
    cv::Mat merge_mat; // merge bg and fg according pha
    bool flag;

    MattingContentType() : flag(false)
    {};
} MattingContent;

class RobustVideoMatting
{
protected:
    Ort::Env ort_env;
    Ort::Session ort_session{ nullptr };
    // CPU MemoryInfo
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // hardcode input node names
    unsigned int num_inputs = 6;
    std::vector<const char *> input_node_names = {
        "src",
        "r1i",
        "r2i",
        "r3i",
        "r4i",
        "downsample_ratio"
    };
    // init dynamic input dims
    std::vector<std::vector<int64_t>> dynamic_input_node_dims = {
        {1, 3, 1280, 720}, // src  (b=1,c,h,w)
        {1, 1, 1,    1}, // r1i
        {1, 1, 1,    1}, // r2i
        {1, 1, 1,    1}, // r3i
        {1, 1, 1,    1}, // r4i
        {1} // downsample_ratio dsr
    }; // (1, 16, ?h, ?w) for inner loop rxi

    // hardcode output node names
    unsigned int num_outputs = 6;
    std::vector<const char *> output_node_names = {
        "fgr",
        "pha",
        "r1o",
        "r2o",
        "r3o",
        "r4o"
    };
    wstring onnx_path;
    //const char *log_id = nullptr;
    bool context_is_update = false;

    // input values handler & init
    std::vector<float> dynamic_src_value_handler;
    std::vector<float> dynamic_r1i_value_handler = {0.0f}; // init 0. with shape (1,1,1,1)
    std::vector<float> dynamic_r2i_value_handler = {0.0f};
    std::vector<float> dynamic_r3i_value_handler = {0.0f};
    std::vector<float> dynamic_r4i_value_handler = {0.0f};
    std::vector<float> dynamic_dsr_value_handler = {0.25f}; // downsample_ratio with shape (1)
    

    // return normalized src, rxi, dsr Tensors
    vector<Ort::Value> transform(const cv::Mat& mat);

public:
    RobustVideoMatting() =delete;
    RobustVideoMatting(wstring _onnx_path, int num_threads);
    void detect(const cv::Mat& mat, MattingContent& content,
        float downsample_ratio = 0.25f, bool video_mode = false);

    void detect_video(const std::string& video_path,
        const std::string& output_path,
        std::vector<MattingContent>& contents,
        bool save_contents = false,
        float downsample_ratio = 0.25f,
        unsigned int writer_fps = 30);

    // data format =0 Ϊchw 
    Ort::Value create_tensor(cv::Mat& mat,vector<int64_t>& tensor_dims,
        const Ort::MemoryInfo& memory_info_handler,
        std::vector<float>& tensor_value_handler,int data_format);

    int64_t value_size_of(vector<int64_t>& dims); // get value size
    void generate_matting(std::vector<Ort::Value>& output_tensors,
        MattingContent& content,cv::Mat raw_image);

    void update_context(std::vector<Ort::Value>& output_tensors);

 };
















