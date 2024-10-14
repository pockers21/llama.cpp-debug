#include "common.h"
#include "llama-impl.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <set>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct llama_sampling_context * ctx_sampling;
};

void llama_sample_log_softmax_impl(float* logits, size_t n_logits) {
    if (n_logits == 0) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // 找到最大的 logit
    float max_logit = logits[0];
    for (size_t i = 1; i < n_logits; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    // 计算 exp 和
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n_logits; ++i) {
        float exp_val = expf(logits[i] - max_logit);
        sum_exp += exp_val;
    }

    // 计算 log(sum(exp))
    float log_sum_exp = logf(sum_exp);

    // 计算 log_softmax
    for (size_t i = 0; i < n_logits; ++i) {
        logits[i] = logits[i] - max_logit - log_sum_exp;
    }

    int64_t t_sample_us = ggml_time_us() - t_start_sample_us;
    std::cout << "Time taken: " << t_sample_us << " microseconds" << std::endl;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_split  = params.p_split;

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    std::default_random_engine rng(params.seed);
    std::uniform_real_distribution<> u_dist;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    llama_init_result llama_init_tgt = llama_init_from_gpt_params(params);
    model_tgt = llama_init_tgt.model;
    ctx_tgt = llama_init_tgt.context;

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.n_threads_draft > 0) {
        params.n_threads = params.n_threads_draft;
    }
    params.n_threads_batch = params.n_threads_batch_draft;
    llama_init_result llama_init_dft = llama_init_from_gpt_params(params);
    model_dft = llama_init_dft.model;
    ctx_dft = llama_init_dft.context;

    const bool vocab_type_tgt = llama_vocab_type(model_tgt);
    LOG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(model_dft);
    LOG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        fprintf(stderr, "%s: error: draft model vocab type must match target model to use speculation but ", __func__);
        fprintf(stderr, "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)
    ) {
        fprintf(stderr, "%s: error: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            fprintf(stderr, "%s: error: draft model vocab must closely match target model to use speculation but ", __func__);
            fprintf(stderr, "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                fprintf(stderr, "%s: error: draft model vocab must match target model to use speculation but ", __func__);
                fprintf(stderr, "token %d content differs - target '%s', draft '%s'\n", i,
                        llama_token_to_piece(ctx_tgt, i).c_str(),
                        llama_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();





    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;


    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    if (params.sparams.temp == 0) {
        params.sparams.temp = -1.0f; // force greedy sampling with probs for the draft model
    }

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
    }
     LLAMA_LOG_INFO("params.n_ctx: \"%d\"\n",params.n_ctx);

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, 1);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, n_seq_dft);

    //mock input param
    const int total_tokens  = 59;
    const int depth = 5;
    const int top_k = 10;
    const int threshold = 1.0
    const int max_new_tokens = 512;
    // TODO assume logits processor is None
    // eval the prompt with both models
    //TODO llama.cpp only allow output logits or hidden_states, we should output both
    //turn on logits_all switch, n_outputs will be setted to n_tokens_all
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input , 0, 0));




    //TODO add sample procedure
    const int sampled_token_id = 0;

    int n_past_tgt = inp.size();
    inp.push_back(sampled_token_id);
    inp.erase(inp.begin());
    //draft model input consist embedding data, so should be inited by llama_batch_init
    //after llama_batch_init, batch_dft.logits and batch_dft.embd would be placed with logits and hidden states generated from target model
    llama_bacth batch_dft = llama_batch_init(n_input, 1, 1);
    llama_decode(ctx_dft, batch_dft);
    const float * logits = ctx_dft.logits;
    const float * embd = ctx_dft.embd;

    //
    std::vector<std::vector<float>> scores_list；
    std::vector<std::vector<int>> parents_list；
    std::vector<std::vector<int>> topk_tokens_list；

    const auto t_dec_start = ggml_time_us();



    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ndraft:\n");
    llama_print_timings(ctx_dft);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    llama_sampling_free(ctx_sampling);
    for (int s = 0; s < n_seq_dft; ++s) {
        llama_sampling_free(drafts[s].ctx_sampling);
    }

    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
