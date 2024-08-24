#include <cstdio>
#include <cstdarg>
#include <cstring>


enum ggml_log_level { GGML_LOG_LEVEL_INFO, GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR };

void llama_log_internal_new(ggml_log_level level, const char *file_name, const char *fun_name, int line_no, const char *format, ...);

static void llama_log_internal_v(ggml_log_level level,
                                const char * format,
                                const char * file_name,
                                const char * fun_name,
                                const int line_no,
                                va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);

    char prefix[256];
    snprintf(prefix, sizeof(prefix), "[%s:%s:%d] ", file_name, fun_name, line_no);

    int prefix_len = strlen(prefix);
    char buffer[128];
    int len = vsnprintf(buffer, sizeof(buffer), format, args);

    if (len + prefix_len < sizeof(buffer)) {
        memmove(buffer + prefix_len, buffer, len + 1);
        memcpy(buffer, prefix, prefix_len);
	//g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len + prefix_len + 1];
        snprintf(buffer2, prefix_len + 1, "%s", prefix);
        vsnprintf(buffer2 + prefix_len, len + 1, format, args_copy);
        //g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }

    va_end(args_copy);
}

void llama_log_internal_new(ggml_log_level level,
                        const char * file_name,
                        const char * fun_name,
                        const int line_no,
                        const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, file_name, fun_name ,line_no, args);
    va_end(args);
}

#define LLAMA_LOG_INFO(...)  llama_log_internal_new(GGML_LOG_LEVEL_INFO , __FILE__, __func__, __LINE__, __VA_ARGS__)

int main() {
    LLAMA_LOG_INFO("Hello, %s!\n", "world");
    return 0;
}

