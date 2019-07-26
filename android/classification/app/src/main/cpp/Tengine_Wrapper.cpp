//
// Created by user on 10/26/2018.
//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include "Tengine_Wrapper.h"
#include "opencv2/imgcodecs.hpp"

//#inlucde "string.h"

#include <android/log.h>
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,"JPY",__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,"JPY",__VA_ARGS__)
#define HEATMAP_SIZE 16

int TengineWrapper::InitTengine()
{
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return -1;

    /* load tf model  */

    /*
    const char* mobilenet_tf_model = "/data/local/tmp/squeezenet.pb";   // mobileNet path: /data/local/tmp/frozen_mobilenet_v1_224.pb    /data/local/tmp/squeezenet.pb 模型名字也有关系  /sdcard/frozen_mobilenet_v1_224.pb
    LOGI("---------predict classification with squeezenet.pb model----------");
    const char* format = "tensorflow";
    LOGI("---------after format string----------");

    if (load_model("squeezenet", format, mobilenet_tf_model) < 0)    // error here    // loadmodel有关系   mobilenet
    {
        LOGI("---------loadmodel() < 0----------");
        return 4;
    }
    else
    {
        LOGI("---------loadmodel() >= 0----------");
    }
    */

    /* load caffe model  */
    const char* mobilenet_caffe_proto = "/data/local/tmp/headPose_r18_7_2_16_64_inputsize128.prototxt";
    const char* mobilenet_caffe_model = "/data/local/tmp/headPose_r18_7_2_16_64_inputsize128.caffemodel";
    const char* format = "caffe";
    LOGI("---------after format string----------");

    if (load_model("headpose", format, mobilenet_caffe_proto, mobilenet_caffe_model) == 0)
    {
        LOGI("---------loadmodel success----------");
    }
    else
    {
        LOGI("---------loadmodel() >= 0----------");
        return 4;
    }


    g_mobilenet = create_runtime_graph("graph0","headpose",NULL);  //mobilenet
    if (!check_graph_valid(g_mobilenet))
        return 5;
    else
    {
        LOGI("---------graph valid----------");
    }

    const int img_h = 128;  // 224
    const int img_w = 128;  // 224

    int image_size = img_h * img_w * 3;
    g_mobilenet_input = (float*) malloc(sizeof(float) * image_size);

    int dims[4] = {1, 3, img_h, img_w};

    tensor_t input_tensor = get_graph_input_tensor(g_mobilenet, 0, 0);
    if(!check_tensor_valid(input_tensor))
        return 6;

    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, g_mobilenet_input, image_size * 4);

    if( prerun_graph(g_mobilenet)!=0 )
        return 1;


    LOGI("---------TengineWrapper::InitTengine() success----------");
    return 0;
}


int TengineWrapper::ReleaseTengine()
{
    sleep(1);
    /*
    tensor_t input_tensor = get_graph_input_tensor(g_mobilenet, 0, 0);
    put_graph_tensor(input_tensor);
    free(g_mobilenet_input);
    postrun_graph(g_mobilenet);
    destroy_runtime_graph(g_mobilenet);
    remove_model("mobilenet");
     */
    return 0;
}

int TengineWrapper::get_input_data(const char* image, float* data, int img_h, int img_w)
{
    cv::Mat sample = cv::imread(image);
    if (sample.empty())
        return 1;
    cv::Mat img;
    if (sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img=sample;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    float mean[3]={104.f,117.f,123.f};
    //float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[c * hw + h * img_w + w] = (*img_data - mean[c])*0.017;
                img_data++;
            }
        }
    }

    LOGI("---------TTengineWrapper::get_input_data(char*image) success----------");
/**/
    return 0;
}

int TengineWrapper::get_input_data(cv::Mat sample, float* data, int img_h, int img_w)
{
    if (sample.empty())
        return 1;
    cv::Mat img;
    if (sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img=sample;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    //float mean[3]={104.f,117.f,123.f};
    float mean[3] = {123.675, 116.280, 103.530};
    float std[3] = {58.395, 57.120, 57.375};
    //float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[c * hw + h * img_w + w] = (*img_data - mean[c])/std[c]; //*0.017;
                img_data++;
            }
        }
    }
    LOGI("---------TTengineWrapper::get_input_data(cv::Mat) success----------");
/**/
    return 0;
}

int TengineWrapper::RunTengine(const char* image)
{
    LOGI("---------run_graph(char* image)----------");

    if( get_input_data(image, g_mobilenet_input, 128, 128) )
        return 7;

    if( !run_graph(g_mobilenet,1))
        return 2;

    LOGI("---------TTengineWrapper::RunTengine(char*image) success----------");
    return 0;
}
int TengineWrapper::RunTengine(cv::Mat sample)
{

    LOGI("---------run_graph(cv::Mat)----------");
    if( get_input_data(sample, g_mobilenet_input, 128, 128) )   // 224, 224
    {
        return 7;
    }
    else
    {
        LOGI("---------get_input_data(sample, g_mobilenet_input, 128, 128) success----------");
    }

    if( !run_graph(g_mobilenet,1))   // run_graph: 0 means success, -1 means fail
    {
        LOGI("---------run_graph(g_mobilenet,1) return 0, success----------");
    }
    else
    {
        LOGI("---------run_graph(g_mobilenet,1) return -1, failure----------");
        return 2;
    }

    LOGI("---------TTengineWrapper::RunTengine(cv::Mat) success----------");
    return 0;
}
std::string TengineWrapper::GetTop1()
{
    LOGI("---------TTengineWrapper::GetTop1() begin----------");
    const char* label_file = "/data/local/tmp/labels.txt";   // synset_words.txt
    std::vector<std::string> result;
    std::ifstream labels(label_file);

    std::string line;
    while (std::getline(labels, line))
        result.push_back(line);

    int true_id = 0;
    float true_score=0.f;

    LOGI("---------get output_tensor begin----------");
    tensor_t output_tensor =  get_graph_tensor(g_mobilenet, "ThresholdBackward98/0");    //get_graph_output_tensor(g_mobilenet, 0, 0);  //  segmentation fault, caused by empty pointer;
    if(NULL == output_tensor)
    {
        LOGI("---------output_tensor empty----------");
    }
    else
    {
        LOGI("---------output_tensor has data----------");
    }

    LOGI("---------get output_tensor after----------");
    float *data = (float *)get_tensor_buffer(output_tensor);
    LOGI("---------get output_tensor success----------");


    /* postrun function: get headpose vector by integral calculation*/
    float headpose[6];
    post_run(data, headpose, HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIZE*2, 2);


    std::string res;
    for(int i=0; i<6; i++)
    {
        res = res + " " +  std::to_string(headpose[i]);
        std::cout << headpose[i] << " ";
    }
    std::cout << std::endl;

    LOGI("---------TTengineWrapper::::GetTop1() success----------");
    return res;  // result[true_id - 1]
}

/*
Name: post_run
Funtion: processing network's output to get headpose vector
  para@ outdata: NN's output tensor 1*32*16*16
  para@ result: headpose vector pointer 1×6（1×3×2, two vector）
  height: HEATMAP_SIZE
  width:  HEATMAP_SIZE
  depth:  HEATMAP_SIZE*2
  dim:    2
*/

void TengineWrapper::post_run(float* outdata, float* result, int height, int width, int depth, int dim)
{

    // ----------1. Calculate softmax  values for pred, including 2 parts---------
    float soft_sum_1 = 0.0;
    float soft_sum_2 = 0.0;
    int loop_num = height*width*depth/2;  // 128×64×64 break into two 64×64×64
    for(int i=0; i<loop_num; i++)
    {
        soft_sum_1 += exp(*(outdata+i));
        soft_sum_2 += exp(*(outdata+loop_num+i));
    }

    float sum_1 = 0.0;
    float sum_2 = 0.0;
    for(int i=0; i<loop_num; i++)
    {
        *(outdata+i) = exp(*(outdata+i)) / soft_sum_1;  // sum for the first part
        *(outdata+loop_num+i) = exp(*(outdata+loop_num+i)) / soft_sum_2;  // sum for the second part
        sum_1 += *(outdata+i);
        sum_2 += *(outdata+loop_num+i);
    }

    // ----------2. Integral pred values-------------
    int depth_half = depth / 2;
    float out_xy[HEATMAP_SIZE][HEATMAP_SIZE] = {0};
    float out_x[HEATMAP_SIZE] = {0};
    float out_1 = 0.0;             //  x for first vector
    float out_xy_2[HEATMAP_SIZE][HEATMAP_SIZE] = {0};
    float out_x_2[HEATMAP_SIZE] = {0};       //  x for second vector
    float out_4 = 0.0; // value integral to width diretion

    for(int i=0; i<width; i++)   // x axis
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<depth_half; k++)
            {
                out_xy[i][j] += *(outdata + k*height*width + j*height + i);
                out_xy_2[i][j] += *(outdata + loop_num + k*height*width + j*height + i);
            }
            out_x[i] += out_xy[i][j];
            out_x_2[i] += out_xy_2[i][j];
        }
        out_1 += i*out_x[i];
        out_4 += i*out_x_2[i];
    }

    float out_xz[HEATMAP_SIZE][HEATMAP_SIZE] = {0};
    float out_y[HEATMAP_SIZE] = {0};
    float out_2 = 0.0;
    float out_xz_2[HEATMAP_SIZE][HEATMAP_SIZE] = {0};
    float out_y_2[HEATMAP_SIZE] = {0};
    float out_5 = 0.0;

    for(int i=0; i<height; i++)  // y axis
    {
        for(int j=0; j<width; j++)
        {
            for(int k=0; k<depth_half; k++)
            {
                out_xz[i][j] += *(outdata + k*height*width + i*height + j);
                out_xz_2[i][j] += *(outdata + loop_num + k*height*width + i*height + j);

            }
            out_y[i] += out_xz[i][j];
            out_y_2[i] += out_xz_2[i][j];
        }
        out_2 += i*out_y[i];
        out_5 += i*out_y_2[i];
    }

    float out_yz[HEATMAP_SIZE][HEATMAP_SIZE]={0};
    float out_z[HEATMAP_SIZE]={0};
    float out_3 = 0.0;
    float out_yz_2[HEATMAP_SIZE][HEATMAP_SIZE]={0};
    float out_z_2[HEATMAP_SIZE]={0};
    float out_6 = 0.0;

    for(int i=0; i<depth_half; i++)  // z axis
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<width; k++)
            {
                out_yz[i][j] += *(outdata + i*height*width + j*height + k);
                out_yz_2[i][j] += *(outdata + loop_num + i*height*width + j*height + k);
            }
            out_z[i] += out_yz[i][j];
            out_z_2[i] += out_yz_2[i][j];
        }
        out_3 += i*out_z[i];
        out_6 += i*out_z_2[i];
    }
    std::cout << "out_1 2 3 " << out_1 << " " << out_2 << " " << out_3 << std::endl;
    std::cout << "out_4 5 6 " << out_4 << " " << out_5 << " " << out_6 << std::endl;


    // ------------3. Normalization--------------
    out_1 = (out_1  - float((width - 1.0) / 2.0)) / float((width - 1.0) / 2.0);
    out_2 = (out_2 - float((height - 1.0) / 2.0)) / float((height - 1.0) / 2.0);
    out_3 = (out_3 - float((depth_half - 1.0) / 2.0)) / float((depth_half - 1.0) / 2.0);
    out_4 = (out_4  - float((width - 1.0) / 2.0)) / float((width - 1.0) / 2.0);
    out_5 = (out_5 - float((height - 1.0) / 2.0)) / float((height - 1.0) / 2.0);
    out_6 = (out_6 - float((depth_half - 1.0) / 2.0)) / float((depth_half - 1.0) / 2.0);
    std::cout << "out_1 2 3(After Normalization) " << out_1 << " " << out_2 << " " << out_3 << std::endl;
    std::cout << "out_4 5 6(After Normalization) " << out_4 << " " << out_5 << " " << out_6 << std::endl;

    // ------------4.Return the result-----------
    *result = out_1;
    *(result+1) = out_2;
    *(result+2) =  out_3;
    *(result+3) =  out_4;
    *(result+4) =  out_5;
    *(result+5) =  out_6;

}