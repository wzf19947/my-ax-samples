/*
* AXERA is pleased to support the open source community by making ax-samples available.
*
* Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSD-3-Clause
*
* Unless required by applicable law or agreed to in writing, software distributed
* under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
* CONDITIONS OF ANY KIND, either express or implied. See the License for the
* specific language governing permissions and limitations under the License.
*/

/*
* Author: ZHEQIUSHUI
*/

#include <cstdio>
#include <cstring>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"
#include "tokenizer/tokenizer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>
#define CLS_TOKEN 101
#define SEP_TOKEN 102
#define PAD_TOKEN 0
#define MAX_TOKENS 512
#define TOKEN_FEATURE_DIM 384

struct embeding_t {
    float embeding[TOKEN_FEATURE_DIM];
    int len_of_tokens;
    embeding_t() {
        memset(embeding, 0, sizeof(embeding)); // ← 必须有！
        len_of_tokens = 0;
    }
};

struct embeding_handle_internal_t
{
    // std::shared_ptr<ax_runner_base> runner;
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;
};
const int DEFAULT_LOOP_COUNT = 1;
typedef void * embeding_handle_t;

namespace ax
{
    int ax_embeding(embeding_handle_internal_t * internal, char *text, embeding_t *embeding, AX_ENGINE_IO_T io_data, AX_ENGINE_HANDLE handle)
    {
        std::vector<int> _token_ids;
        _token_ids = internal->tokenizer->encode(text);
        if (_token_ids.size() > MAX_TOKENS)
        {
            fprintf(stderr, "text len %d > MAX_TOKENS %d, truncate to %d", _token_ids.size(), MAX_TOKENS, MAX_TOKENS);
            _token_ids.resize(MAX_TOKENS);
        }

        _token_ids.insert(_token_ids.begin(), CLS_TOKEN);
        _token_ids.push_back(SEP_TOKEN);
        memset(io_data.pInputs[0].pVirAddr, 0, io_data.pInputs[0].nSize);
        memcpy(io_data.pInputs[0].pVirAddr, _token_ids.data(), _token_ids.size() * sizeof(int));

        // 9. run model
        int ret = AX_ENGINE_RunSync(handle, &io_data);
        SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
        AX_SYS_MinvalidateCache(io_data.pOutputs[0].phyAddr, io_data.pOutputs[0].pVirAddr, io_data.pOutputs[0].nSize);
        embeding->len_of_tokens = _token_ids.size();
        memcpy(embeding->embeding, (float*)io_data.pOutputs[0].pVirAddr, TOKEN_FEATURE_DIM * sizeof(float));

        return 0;
    }

    float ax_similarity(const embeding_t *embeding_1, const embeding_t *embeding_2)
    {
        if (embeding_1 == nullptr || embeding_2 == nullptr) return -1;

        // 计算归一化后的相似度，但不修改原数组
        float norm1 = 0.0f, norm2 = 0.0f;
        for (int i = 0; i < TOKEN_FEATURE_DIM; i++) {
            norm1 += embeding_1->embeding[i] * embeding_1->embeding[i];
            norm2 += embeding_2->embeding[i] * embeding_2->embeding[i];
        }
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);

        float sim = 0.0f;
        for (int i = 0; i < TOKEN_FEATURE_DIM; i++) {
            sim += (embeding_1->embeding[i] / norm1) * (embeding_2->embeding[i] / norm2);
        }
        
        sim = sim < 0 ? 0 : sim > 1 ? 1 : sim;
        return sim;
    }

    bool run_model(const std::string& model, const std::string& tokenizer_model, const int& repeat)
    {
        // 1. init engine

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        auto ret = AX_ENGINE_Init(&npu_attr);

        if (0 != ret)
        {
            return ret;
        }

        // 2. load model
        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer))
        {
            fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
            return false;
        }

        // 3. create handle
        AX_ENGINE_HANDLE handle;
        ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating handle is done.\n");

        // 4. create context
        ret = AX_ENGINE_CreateContext(handle);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating context is done.\n");

        // 5. set io
        AX_ENGINE_IO_INFO_T* io_info;
        ret = AX_ENGINE_GetIOInfo(handle, &io_info);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine get io info is done. \n");
        middleware::print_io_info(io_info);

        // 6. alloc io
        AX_ENGINE_IO_T io_data;
        ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine alloc io is done. \n");
        // 7. insert input
        embeding_handle_internal_t *internal = new embeding_handle_internal_t();
        internal->tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_model));
        if (internal->tokenizer == nullptr)
        {
            fprintf(stdout, "create tokenizer failed");
            return -1;
        }
        std::vector<std::string> sentences_1 = {"I really love math", "so do I"};
        std::vector<std::string> sentences_2 = {"I pretty like mathematics", "same as me"};

        for(int i =0;i<sentences_1.size();i++)
        {
            embeding_t embeding1 = embeding_t();
            embeding_t embeding2 = embeding_t();
            ax_embeding(internal, (char *)sentences_1[i].c_str(), &embeding1, io_data, handle);
            for(int j=0;j<sentences_2.size();j++)
            {
                ax_embeding(internal, (char *)sentences_2[j].c_str(), &embeding2, io_data, handle);
                float sim = ax_similarity(&embeding1, &embeding2);
                printf("similarity between \33[32m%s\33[0m and \33[34m%s\33[0m is %f\n", sentences_1[i].c_str(), sentences_2[j].c_str(), sim);
            }
        }
        fprintf(stdout, "--------------------------------------\n");
        middleware::free_io(&io_data);
        return AX_ENGINE_DestroyHandle(handle);
    }
} // namespace ax

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("token", 't', "token file", false, "./bge_tokenizer.txt");
    // cmd.add<std::string>("input", 'i', "input text", true, "");

    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto token_file = cmd.get<std::string>("token");
    // auto input_text = cmd.get<std::string>("input");

    auto model_file_flag = utilities::file_exist(model_file);
    auto token_file_flag = utilities::file_exist(token_file);

    if (!model_file_flag | !token_file_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };

        if (!model_file_flag) { show_error("model", model_file); }
        if (!token_file_flag) { show_error("token", token_file); }

        return -1;
    }


    auto repeat = cmd.get<int>("repeat");

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "token file : %s\n", token_file.c_str());
    // fprintf(stdout, "input_text : %s\n", input_text.c_str());
    fprintf(stdout, "--------------------------------------\n");


    // 3. sys_init
    AX_SYS_Init();

    // 4. -  engine model  -  can only use AX_ENGINE** inside
    {
        // AX_ENGINE_NPUReset(); // todo ??
        // ax::run_model(model_file, token_file, repeat, input_text);
        ax::run_model(model_file, token_file, repeat);

        // 4.3 engine de init
        AX_ENGINE_Deinit();
        // AX_ENGINE_NPUReset();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}
