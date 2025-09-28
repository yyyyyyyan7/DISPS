#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>
#include <string>
#include <mutex>
#include <cassert>
#include <chrono>
#include "Cluster.cpp"
#include "topic.cpp"

namespace Config {
    constexpr int DIM = 1024;
    constexpr float ALPHA = 0.66f;
    constexpr float BETA = 0.76f;
    constexpr int CHECK = 10;
    constexpr bool TEST_MODE = false;
    constexpr int PUB_START_IDX = 1;
}

// Global objects
TOPIC_clu topic_clu;
TOPIC_index topic_index;
std::mutex mtx_topic;
std::mutex mtx_clu;

/**
 * @brief Reads text and embedding data from given files.
 * Each line in text_file contains whitespace-separated tokens.
 * Each line in embeddings_file contains comma-separated floats.
 *
 * @param text_file_path Path to text data file.
 * @param embeddings_file_path Path to embedding vectors file.
 * @param text_data Output container for tokenized text lines.
 * @param embeddings_data Output container for float embedding vectors.
 * @return true if both files are successfully read; false otherwise.
 */
bool read_data(const std::string& text_file_path,
               const std::string& embeddings_file_path,
               std::vector<std::vector<std::string>>& text_data,
               std::vector<std::vector<float>>& embeddings_data) {
    std::ifstream text_file(text_file_path);
    std::ifstream embeddings_file(embeddings_file_path);

    if (!text_file.is_open()) {
        std::cerr << "Error: Cannot open text file: " << text_file_path << std::endl;
        return false;
    }
    if (!embeddings_file.is_open()) {
        std::cerr << "Error: Cannot open embeddings file: " << embeddings_file_path << std::endl;
        return false;
    }

    std::string line_t, line_e;
    while (std::getline(text_file, line_t) && std::getline(embeddings_file, line_e)) {
        // Tokenize text line by whitespace
        std::istringstream iss_t(line_t);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss_t}, std::istream_iterator<std::string>{}};
        text_data.push_back(std::move(tokens));

        // Parse embedding vector from comma-separated floats
        std::vector<float> embedding;
        std::istringstream iss_e(line_e);
        std::string val;
        while (std::getline(iss_e, val, ',')) {
            try {
                embedding.push_back(std::stof(val));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse float from '" << val << "' - " << e.what() << std::endl;
            }
        }
        embeddings_data.push_back(std::move(embedding));
    }

    return true;
}

int main(int argc, char* argv[]) {
    // Allow file paths from command line arguments, otherwise use defaults
    const std::string text_file_path = (argc > 1) ? argv[1] : "dataset/short_texts.txt";
    const std::string embeddings_file_path = (argc > 2) ? argv[2] : "dataset/short_vectors.csv";

    std::cout << "Starting clustering program..." << std::endl;
    std::cout << "Text file: " << text_file_path << std::endl;
    std::cout << "Embeddings file: " << embeddings_file_path << std::endl;

    // Initialize clustering and topic index with configuration parameters
    topic_clu.iniDimandThres(Config::DIM, Config::ALPHA, Config::TEST_MODE, Config::CHECK);
    topic_index.ini(Config::BETA, Config::TEST_MODE);

    std::vector<std::vector<std::string>> text_data;
    std::vector<std::vector<float>> embeddings_data;

    // Load data from files
    if (!read_data(text_file_path, embeddings_file_path, text_data, embeddings_data)) {
        std::cerr << "Failed to read input data. Exiting." << std::endl;
        return -1;
    }
    if (text_data.size() != embeddings_data.size()) {
        std::cerr << "Mismatch between number of text lines and embeddings." << std::endl;
        return -1;
    }

    std::cout << "Loaded " << text_data.size() << " samples." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    int pub_count = 0;
    int sub_count = 0;
    int SUB_id = 0;
    bool publish_mode = false;

    // Main processing loop: first subscription queries, then publishing
    for (size_t i = 0; i < text_data.size(); ++i) {
        const auto& embedding = embeddings_data[i];
        const auto& text_line = text_data[i];

        if (publish_mode) {
            ++pub_count;

            // Attempt to find an existing topic for the embedding
            int topic_found = topic_index.find_topic_HNSW(embedding, text_line);
            if (topic_found < 0) {
                // If no topic found, add embedding to clustering structure
                topic_clu.online_add(embedding, topic_index, text_line);
            }

            if (pub_count >= 2000000) {
                std::cout << "Reached maximum publish count limit. Exiting loop." << std::endl;
                break;
            }

        } else {
            ++sub_count;

            // Process subscription queries (top-K topic search)
            topic_index.find_topK_topic(embedding, SUB_id++);
            if (sub_count >= 1000000) {
                publish_mode = true;
                // Reset index to start publishing from configured start index
                i = Config::PUB_START_IDX - 1; // Because for loop increments i after continue
                std::cout << "Switching to publish mode after " << sub_count << " subscription queries." << std::endl;
            }
        }
    }

    // Finalize clustering by adjusting topics to nearest clusters
    topic_clu.go_to_nearest_topic(topic_index, 100);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total processing time: " << elapsed.count() << " seconds." << std::endl;

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}

