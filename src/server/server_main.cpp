#include "qwen3_tts.h"
#include "server/wav_memory.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace {

struct server_options {
    std::string model_dir;
    std::string model_name;
    std::string host     = "127.0.0.1";
    int         port     = 8080;
    int         threads  = 4;
};

void print_usage(const char * program) {
    fprintf(stderr,
            "Usage: %s -m <model_dir> [options]\n"
            "\n"
            "Options:\n"
            "  -m, --model <dir>        Model directory (required)\n"
            "      --model-name <name>  Optional base name for model files\n"
            "      --host <addr>        Bind address (default: 127.0.0.1)\n"
            "      --port <n>           Port (default: 8080)\n"
            "  -j, --threads <n>        Default synthesis threads (default: 4)\n"
            "  -h, --help               Show this help\n",
            program);
}

bool parse_args(int argc, char ** argv, server_options & opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing value for %s\n", name);
                std::exit(1);
            }
            return argv[i];
        };
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "-m" || a == "--model") {
            opt.model_dir = need("--model");
        } else if (a == "--model-name") {
            opt.model_name = need("--model-name");
        } else if (a == "--host") {
            opt.host = need("--host");
        } else if (a == "--port") {
            opt.port = std::atoi(need("--port"));
        } else if (a == "-j" || a == "--threads") {
            opt.threads = std::atoi(need("--threads"));
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (opt.model_dir.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

int language_to_id(const std::string & lang, bool & ok) {
    ok = true;
    if (lang == "en" || lang == "english")       return 2050;
    if (lang == "ru" || lang == "russian")       return 2069;
    if (lang == "zh" || lang == "chinese")       return 2055;
    if (lang == "ja" || lang == "japanese")      return 2058;
    if (lang == "ko" || lang == "korean")        return 2064;
    if (lang == "de" || lang == "german")        return 2053;
    if (lang == "fr" || lang == "french")        return 2061;
    if (lang == "es" || lang == "spanish")       return 2054;
    if (lang == "it" || lang == "italian")       return 2070;
    if (lang == "pt" || lang == "portuguese")    return 2071;
    ok = false;
    return 2050;
}

// Apply JSON-provided param fields onto a tts_params struct.
// Returns false with err set on invalid input.
bool apply_params_from_json(const json & j, qwen3_tts::tts_params & p,
                            int default_threads, std::string & err) {
    p.n_threads     = default_threads;
    p.print_progress = false;
    p.print_timing   = false;

    if (j.contains("temperature"))        p.temperature        = j["temperature"].get<float>();
    if (j.contains("top_p"))              p.top_p              = j["top_p"].get<float>();
    if (j.contains("top_k"))              p.top_k              = j["top_k"].get<int32_t>();
    if (j.contains("max_tokens"))         p.max_audio_tokens   = j["max_tokens"].get<int32_t>();
    if (j.contains("repetition_penalty")) p.repetition_penalty = j["repetition_penalty"].get<float>();
    if (j.contains("threads"))            p.n_threads          = j["threads"].get<int32_t>();
    if (j.contains("instruction"))        p.instruction        = j["instruction"].get<std::string>();
    if (j.contains("speaker"))            p.speaker            = j["speaker"].get<std::string>();

    if (j.contains("language")) {
        const auto & lang_field = j["language"];
        if (lang_field.is_number_integer()) {
            p.language_id = lang_field.get<int32_t>();
        } else if (lang_field.is_string()) {
            bool ok;
            int id = language_to_id(lang_field.get<std::string>(), ok);
            if (!ok) {
                err = "unknown language: " + lang_field.get<std::string>();
                return false;
            }
            p.language_id = id;
        } else {
            err = "language must be a string code or integer id";
            return false;
        }
    }
    return true;
}

std::string random_suffix() {
    static std::mt19937_64 rng{std::random_device{}()};
    uint64_t v = rng();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

bool write_tempfile(const std::string & bytes, fs::path & out_path) {
    fs::path dir   = fs::temp_directory_path();
    out_path       = dir / ("qwen3tts_" + random_suffix() + ".wav");
    std::ofstream f(out_path, std::ios::binary);
    if (!f) return false;
    f.write(bytes.data(), (std::streamsize) bytes.size());
    f.close();
    return bool(f);
}

void send_json(httplib::Response & res, int status, const json & j) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

void send_error(httplib::Response & res, int status, const std::string & msg) {
    send_json(res, status, json{{"error", msg}});
}

void send_wav(httplib::Response & res, const std::vector<float> & audio, int sample_rate,
              const qwen3_tts::tts_result & r) {
    std::vector<uint8_t> wav;
    if (!qwen3_tts::wav_encode(audio, sample_rate, wav)) {
        send_error(res, 500, "wav_encode failed");
        return;
    }
    char dur[32];
    std::snprintf(dur, sizeof(dur), "%.3f", sample_rate > 0 ? (double) audio.size() / sample_rate : 0.0);
    res.set_header("X-Audio-Duration-Seconds", dur);
    res.set_header("X-Synth-Total-Ms",    std::to_string(r.t_total_ms));
    res.set_header("X-Synth-Tokenize-Ms", std::to_string(r.t_tokenize_ms));
    res.set_header("X-Synth-Encode-Ms",   std::to_string(r.t_encode_ms));
    res.set_header("X-Synth-Generate-Ms", std::to_string(r.t_generate_ms));
    res.set_header("X-Synth-Decode-Ms",   std::to_string(r.t_decode_ms));
    res.status = 200;
    res.set_content(std::string(wav.begin(), wav.end()), "audio/wav");
}

struct synth_context {
    qwen3_tts::Qwen3TTS & tts;
    std::mutex &          mu;
    int                   default_threads;
};

void handle_synthesize(synth_context & ctx, const httplib::Request & req, httplib::Response & res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        send_error(res, 400, std::string("invalid JSON: ") + e.what());
        return;
    }
    if (!body.contains("text") || !body["text"].is_string()) {
        send_error(res, 400, "missing 'text' (string)");
        return;
    }
    std::string text = body["text"].get<std::string>();
    qwen3_tts::tts_params params;
    std::string err;
    if (!apply_params_from_json(body, params, ctx.default_threads, err)) {
        send_error(res, 400, err);
        return;
    }

    std::lock_guard<std::mutex> lock(ctx.mu);
    qwen3_tts::tts_result r;
    if (body.contains("embedding") && body["embedding"].is_array()) {
        std::vector<float> emb;
        emb.reserve(body["embedding"].size());
        for (const auto & v : body["embedding"]) {
            emb.push_back(v.get<float>());
        }
        r = ctx.tts.synthesize_with_speaker_embedding(text, emb, params);
    } else {
        r = ctx.tts.synthesize(text, params);
    }
    if (!r.success) {
        send_error(res, 500, r.error_msg);
        return;
    }
    send_wav(res, r.audio, r.sample_rate, r);
}

const httplib::FormData * find_file(const httplib::Request & req, const std::string & name) {
    auto it = req.form.files.find(name);
    if (it == req.form.files.end()) return nullptr;
    return &it->second;
}

bool parse_params_from_form(const httplib::Request & req,
                            qwen3_tts::tts_params & params,
                            std::string & text, std::string & err, int default_threads) {
    json j = json::object();

    auto text_it = req.form.fields.find("text");
    if (text_it != req.form.fields.end()) {
        text = text_it->second.content;
    }

    auto params_it = req.form.fields.find("params");
    if (params_it != req.form.fields.end()) {
        try {
            j = json::parse(params_it->second.content);
        } catch (const std::exception & e) {
            err = std::string("invalid params JSON: ") + e.what();
            return false;
        }
    }

    for (const auto & kv : req.form.fields) {
        const std::string & name = kv.first;
        if (name == "text" || name == "params") continue;
        j[name] = kv.second.content;
    }

    // numeric coercions for plain form fields
    for (const char * k : {"temperature", "top_p", "repetition_penalty"}) {
        if (j.contains(k) && j[k].is_string()) j[k] = std::stof(j[k].get<std::string>());
    }
    for (const char * k : {"top_k", "max_tokens", "threads"}) {
        if (j.contains(k) && j[k].is_string()) j[k] = std::stoi(j[k].get<std::string>());
    }
    return apply_params_from_json(j, params, default_threads, err);
}

void handle_synthesize_with_voice(synth_context & ctx, const httplib::Request & req, httplib::Response & res) {
    const auto * ref = find_file(req, "reference_audio");
    if (!ref) {
        send_error(res, 400, "missing 'reference_audio' file part");
        return;
    }
    std::string text;
    qwen3_tts::tts_params params;
    std::string err;
    if (!parse_params_from_form(req, params, text, err, ctx.default_threads)) {
        send_error(res, 400, err);
        return;
    }
    if (text.empty()) {
        send_error(res, 400, "missing 'text' form field");
        return;
    }

    fs::path tmp;
    if (!write_tempfile(ref->content, tmp)) {
        send_error(res, 500, "failed to write tempfile for reference audio");
        return;
    }
    qwen3_tts::tts_result r;
    {
        std::lock_guard<std::mutex> lock(ctx.mu);
        r = ctx.tts.synthesize_with_voice(text, tmp.string(), params);
    }
    std::error_code ec;
    fs::remove(tmp, ec);

    if (!r.success) {
        send_error(res, 500, r.error_msg);
        return;
    }
    send_wav(res, r.audio, r.sample_rate, r);
}

void handle_speaker_embedding(synth_context & ctx, const httplib::Request & req, httplib::Response & res) {
    const auto * ref = find_file(req, "reference_audio");
    if (!ref) {
        send_error(res, 400, "missing 'reference_audio' file part");
        return;
    }
    fs::path tmp;
    if (!write_tempfile(ref->content, tmp)) {
        send_error(res, 500, "failed to write tempfile for reference audio");
        return;
    }
    std::vector<float> embedding;
    int64_t encode_ms = 0;
    bool ok;
    {
        std::lock_guard<std::mutex> lock(ctx.mu);
        ok = ctx.tts.extract_speaker_embedding(tmp.string(), embedding, &encode_ms);
    }
    std::error_code ec;
    fs::remove(tmp, ec);
    if (!ok) {
        send_error(res, 500, ctx.tts.get_error());
        return;
    }
    json out = {
        {"embedding",   embedding},
        {"dim",         embedding.size()},
        {"encode_ms",   encode_ms},
    };
    send_json(res, 200, out);
}

void handle_capabilities(qwen3_tts::Qwen3TTS & tts, httplib::Response & res) {
    auto caps = tts.get_model_capabilities();
    json out = {
        {"loaded",                   caps.loaded},
        {"model_type",               caps.model_type},
        {"supports_voice_clone",     caps.supports_voice_clone},
        {"supports_named_speakers",  caps.supports_named_speakers},
        {"supports_instruction",     caps.supports_instruction},
        {"speaker_embedding_dim",    caps.speaker_embedding_dim},
        {"speaker_count",            caps.speaker_count},
    };
    send_json(res, 200, out);
}

void handle_speakers(qwen3_tts::Qwen3TTS & tts, httplib::Response & res) {
    send_json(res, 200, json{{"speakers", tts.get_available_speakers()}});
}

} // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    server_options opt;
    if (!parse_args(argc, argv, opt)) return 1;

    qwen3_tts::Qwen3TTS tts;
    fprintf(stderr, "Loading models from: %s\n", opt.model_dir.c_str());
    auto t0 = std::chrono::steady_clock::now();
    if (!tts.load_models(opt.model_dir, opt.model_name)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "Models loaded in %lld ms\n",
            (long long) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    std::mutex synth_mu;
    synth_context ctx{tts, synth_mu, opt.threads};

    httplib::Server svr;
    svr.set_payload_max_length(256 * 1024 * 1024); // 256 MiB uploads cap

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json models = json::object();
        if (!tts.get_tts_model_path().empty()) {
            models["tts"] = fs::path(tts.get_tts_model_path()).filename().string();
        }
        if (!tts.get_decoder_model_path().empty()) {
            models["decoder"] = fs::path(tts.get_decoder_model_path()).filename().string();
        }
        send_json(res, 200, json{
            {"status",       "ok"},
            {"model_loaded", tts.is_loaded()},
            {"models",       models},
        });
    });
    svr.Get("/v1/capabilities", [&](const httplib::Request &, httplib::Response & res) {
        handle_capabilities(tts, res);
    });
    svr.Get("/v1/speakers", [&](const httplib::Request &, httplib::Response & res) {
        handle_speakers(tts, res);
    });
    svr.Post("/v1/synthesize", [&](const httplib::Request & req, httplib::Response & res) {
        handle_synthesize(ctx, req, res);
    });
    svr.Post("/v1/synthesize_with_voice", [&](const httplib::Request & req, httplib::Response & res) {
        handle_synthesize_with_voice(ctx, req, res);
    });
    svr.Post("/v1/speaker_embedding", [&](const httplib::Request & req, httplib::Response & res) {
        handle_speaker_embedding(ctx, req, res);
    });

    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string msg = "internal error";
        try { if (ep) std::rethrow_exception(ep); }
        catch (const std::exception & e) { msg = e.what(); }
        catch (...) { }
        send_error(res, 500, msg);
    });

    fprintf(stderr, "qwen3-tts-server listening on http://%s:%d\n", opt.host.c_str(), opt.port);
    if (!svr.listen(opt.host, opt.port)) {
        fprintf(stderr, "Error: failed to bind %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    return 0;
}
