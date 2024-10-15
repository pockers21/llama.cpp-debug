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

void get_topk(  const float* probs,
                const int seq_len,
                const int vocab_size,
                int top_k,
                std::vector<int>& indices,
                std::vector<float>& values) {
    // 清空输出 vector
    indices.clear();
    values.clear();

    // 预分配内存
    indices.reserve(seq_len * top_k);
    values.reserve(seq_len * top_k);
    for(size_t i = 0; i < seq_len; i++) {
        std::vector<std::pair<int, float>> candidates(vocab_size);
        for (int j = 0; j < vocab_size; ++j) {
            candidates[j] = {j, probs[j + i*vocab_size]};
        }

        // 部分排序，选出前top_k个最大的元素
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + top_k,
            candidates.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;  // 比较浮点数值
            }
        );

        // 填充结果
        for (int i = 0; i < top_k; ++i) {
            indices.push_back(candidates[i].first);
            values.push_back(candidates[i].second);
        }
    }

}

std::vector<float> create_eye_tensor(int64_t top_k) {
    std::vector<float> result(top_k * top_k, 0.0f);

    for (int64_t i = 0; i < top_k; ++i) {
        result[i * top_k + i] = 1.0f;
    }

    return result;
}

int calulate_bias(const int index, const int topk){
    int bias1 = bias2 = 0;
    if(index > 0){
        bias1 = topk;
    }
    bias = std::max(0, index - 1);
    return 1 + topk * topk * bias2 + bias1;
}

template<typename T>
void flatten_2d_vector(const vector<T>& orgin_vectors,
                       const std::vector<std::vector<T>>& flatted_vector) {
    std::vector<float> flattened;

    size_t total_size = std::accumulate(orgin_vectors.begin(), orgin_vectors.end(), 0,
        [](size_t sum, const std::vector<float>& inner) { return sum + inner.size(); });

    flatted_vector.reserve(total_size);

    for (const auto& inner_vec : orgin_vectors) {
        flatted_vector.insert(flatted_vector.end(), inner_vec.begin(), inner_vec.end());
    }
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
    float * target_model_logits = ctx_tgt.logits;
    float * target_model_embd = ctx_tgt.embd;


    //TODO add sample procedure
    const int sampled_token_id = 0;

    inp.push_back(sampled_token_id);
    inp.erase(inp.begin());

    //draft model input consist embedding data, so should be inited by llama_batch_init
    //after llama_batch_init, batch_dft.logits and batch_dft.embd would be placed with logits and hidden states generated from target model
    const int hidden_dim =  ctx_dft.model->hparams.n_embd;
    llama_bacth batch_dft = llama_batch_init(n_input, hidden_dim, 1);
    int n_past_dft = 0;
    //assign input ids  to batch_dft
    for (size_t i = 0; i < n_input; ++i) {
        llama_batch_add(batch_dft, inp[i], n_past_dft++, { 0 }, true);
    }
    //assign hidden_states to batch_dft
    std::copy(target_model_embd，target_model_embd + n_input * hidden_dim, batch_dft.embd);


    llama_decode(ctx_dft, batch_dft);
    float * draft_logits = ctx_dft.logits;
    float * draft_embd = ctx_dft.embd;
    const int vocab_size = ctx_dft.model->hparams.n_vocab

    std::vector<std::vector<float>> scores_list；
    std::vector<std::vector<int>> parents_list；
    std::vector<std::vector<int>> ss_token；

    //last_p = self.logsoftmax(last_headout)
    const int last_logits_bias = (n_input - 1) * vocab_size;
    const float * last_logits = draft_logits + last_logits_bias;
    llama_sample_log_softmax_impl(last_logits, vocab_size);

    std::vector<int> topk_indices;
    std::vector<float> topk_values;
    get_topk(last_logits, 1, vocab_size, top_k, topk_indices, topk_values);
    ss_token.append(topk_indices);

    //after topk select, topk_indices would be new input_ids
    for (size_t i = 0; i < indices.size(); ++i) {
        std::cout << "Index: " << indices[i]
                  << ", Value: " << values[i] << std::endl;
    }



    std::vector<float> scores  = topk_values;
    scores_list.push_back(scores);

    //
    parents_list.push_back(std::vector<int>(1，1));

    vector<float> repeatted_hidden_states(top_k *  hidden_dim);

    //copy last dimension of hidden_states to repeatted_hidden_states and repeat it
    size_t last_batch_start = (n_input - 1) * hidden_dim;
    std::copy(draft_embd + last_batch_start, draft_embd + last_batch_start + hidden_dim, repeatted_hidden_states.begin());
    for (int i = 1; i < top_k; ++i) {
        std::copy(repeatted_hidden_states.begin(), repeatted_hidden_states.begin() + hidden_dim,
                  repeatted_hidden_states.begin() + i * hidden_dim);
    }
    vector<int32_t> origin_tree_mask = create_eye_tensor(top_k);
    vector<int32_t> tree_mask = origin_tree_mask;


    std::vector<int32_t> topk_cs_index(top_k);
    std::generate(topk_cs_index.begin(), topk_cs_index.end(), [&i]() { return i++; });


    std::vector<int32_t> position_ids(top_k, 0);
    vector<int> inputs_ids = topk_indices;

    for(size_t i = 0; i < depth; i ++) {
        llama_bacth batch_dft = llama_batch_init(top_k, hidden_dim, 1);
        //assign input ids  to batch_dft
        for (size_t i = 0; i < top_k; ++i) {
            llama_batch_add(batch_dft, inputs_ids[i], n_past_dft++, { 0 }, true);
        }
        //assign repeatted hidden_states to batch_dft
        std::copy(repeatted_hidden_states，repeatted_hidden_states + top_k * hidden_dim, batch_dft.embd);
        llama_decode(ctx_dft, batch_dft);

        int bias = calulate_bias(i, top_k);

        std::vector<int32_t> parents = topk_cs_index;
        std::for_each(parents.begin(), parents.end(), [](int& x) { x += bias; });
        parents_list.push_back(parents);

        float * draft_logits = ctx_dft.logits;
        float * draft_embd = ctx_dft.embd;
        for(size_t i = 0; i < top_k; i ++){
            llama_sample_log_softmax_impl(draft_logits + (i * vocab_size), vocab_size);
        }


        std::vector<int> topk_indices;
        std::vector<float> topk_values;
        get_topk(draft_logits, top_k, vocab_size, top_k, topk_indices, topk_values);

        //cu_scores = topk_p + scores[:, None]
        vector<float> cur_scores(top_k * top_k, 0);
        for(size_t index = 0; index < top_k; index ++){
            auto start_pos = cur_scores.begin() + index * top_k;
            auto end_pos = cur_scores.begin() + (index + 1) * top_k;
            std::for_each(start_pos, end_pos,
              [scores[index]](int& element) { element += scores[index]; });
        }

        std::vector<int> topk_cs_indices;
        std::vector<float> topk_cs_values;
        get_topk(&cur_scores[0], 1, top_k, top_k, topk_cs_indices, topk_cs_values);
        scores = topk_cs_values;

        vector<int> out_ids(top_k);
        std::for_each(out_ids.begin(), out_ids.end(),
              [top_k](int& element) { element /= top_k; });

        for(size_t index = 0; index < top_k; index ++){
            auto element = out_ids[index];
            auto src_start_pos = draft_embd + (element * hidden_dim)
            auto src_end_pos = draft_embd + ((element + 1) * hidden_dim)
            auto dst_start_pos = repeatted_hidden_states.begin() + (index * hidden_dim);
            std::copy(src_start_pos, src_end_pos, hidden_dim, dst_start_pos);
        }

        for(size_t index = 0; index < top_k; index ++){
            auto topk_cs_indices_value = topk_cs_indices[index];
            auto element = topk_indices[topk_cs_indices_value];
            inputs_ids.clear();
            inputs_ids.push_back(element);
        }

        ss_token.append(topk_indices);
        scores_list.append(cur_scores);

        //translate tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        const current_expanded_tree_mask_cols = (i+2)*top_k
        const current_expanded_tree_mask_concat_offset = (i+1)*top_k
        vector <int32_t> expanded_tree_mask(top_k, current_expanded_tree_mask_cols);
        for(size_t index = 0; index < top_k; index ++){
            auto element = out_ids[index];
            auto src_start_pos = tree_mask.begin() + element*top_k
            auto src_end_pos = tree_mask.begin() + (element+1)*top_k
            auto dst_start_pos = expanded_tree_mask.begin() + (index * current_expanded_tree_mask_cols);
            std::copy(src_start_pos, src_end_pos, top_k, dst_start_pos);

            src_start_pos = tree_mask.begin() + index*top_k
            src_end_pos = tree_mask.begin() + (index+1)*top_k
            dst_start_pos = expanded_tree_mask.begin() + (index * current_expanded_tree_mask_concat_offset);
            std::copy(src_start_pos, src_end_pos, top_k, dst_start_pos);
        }

    //flatten scores_list
    vector<float> flatted_score_list;
    flatten_2d_vector(scores_list, flatted_score_list);

    //flatten ss_token_list
    vector<int> flatted_ss_token;
    flatten_2d_vector(ss_token, flatted_ss_token);

    std::vector<int> topk_scores_indices;
    std::vector<float> topk_scores_values;
    get_topk(&flatted_score_list[0], 1, top_k, top_k, topk_scores_indices, topk_scores_values);
    std::sort(topk_scores_indices.begin(), topk_scores_indices.end());

    vector<int> draft_tokens;
    std::for_each(topk_scores_indices.begin(), topk_scores_indices.end(),
              [draft_tokens](int& element) { draft_tokens.push_back(element); });

    topk_scores_indices.insert(topk_scores_indices.begin(), sampled_token_id);
    


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
