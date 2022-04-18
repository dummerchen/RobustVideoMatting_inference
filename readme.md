

## 安装与部署

### 通过C++可执行文件运行

环境windows x64

#### 解压release.zip

```bash
unzip Release.zip -d ./
cd Release
```



#### 查看参数帮助

```bash
rvm.exe --help
----------------------------------------
-downsample_ratio (downsample ratio) type: double default: 0.20
-num_thread (threads nums) type: int32 default: 6
-output_path (output path) tyype:string default: "./processed_1.png"
-onnx_path (model path) type: string default: "./onnx/epoch-0.onnx"
-test_path (test path) type: string default: "./1.png"
-rgb (rgb) type:bool default:false

```

* downsample_ratio: 图片下采样倍率，0.25、0.2、0.125, 倍率越小速度越快精度越低，建议使用默认为0.2。
* num_thread:  推理使用的threads数量，效率与所使用cpu有关，理想运行状态是cpu利用90%，并非数量越多越好。
* onnx_path：后缀为.onnx的模型权重文件路径，用于模型推理。
* test_path: 单个文件的测试图片或文件路径，可自动识别视频或图片，输出为当前目录下相应的processed文件。
* output_path: 文件输出路径，要求输出与输入格式对应
* rgb: 是否输出彩色图片或视频，默认输出mask



#### 使用样例

默认参数

```bash
rvm.exe
```

通过自定义路径可以自己选择推理如

推理视频同时改变下采样率

```bash
rvm.exe --downsample_ratio 0.125 --num_thread 6 --test_path ./TEST_01.mp4 --onnx_path ./onnx/epoch-0.onnx
```

推理图片

```bash
rvm.exe --downsample_ratio 0.125 --num_thread 6 --test_path ./1.png --onnx_path ./onnx/epoch-0.onnx
```
