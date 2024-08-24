#pragma once

#define LLAMA_API_INTERNAL
#include "llama.h"

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

//LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
LLAMA_ATTRIBUTE_FORMAT(5,6)
void llama_log_internal_new        (ggml_log_level level, const char *  file_name, const char * fun_name, const int line_no, const char * format, ...);

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal_new(GGML_LOG_LEVEL_INFO , __FILE__, __func__, __LINE__, __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal_new(GGML_LOG_LEVEL_WARN , __FILE__, __func__, __LINE__, __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal_new(GGML_LOG_LEVEL_ERROR, __FILE__, __func__, __LINE__, __VA_ARGS__)

//
// helpers
//

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return; // Avoid infinite loop if 'search' is an empty string
    }
    size_t pos = 0;
    while ((pos = s.find(search, pos)) != std::string::npos) {
        s.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}
