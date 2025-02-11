
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <cmath>
#include <iostream>
#include "Cluster.cpp"
#include "topic.cpp"
#include <iterator>
#include <future>
#include <vector>
#include <string>
#include <mutex>
#include <cassert>
#include <chrono>

TOPIC_clu topic_clu;          
TOPIC_index topic_index;
int D = 1024;  // dim 
bool test = false;
int check = 10;
int SUB_id = 0;

///////////////////////////////////////
float alpha = 0.66;                     
float beta = 8.88889;                  
float threshold_for_topic = 0.802;      
float merge_threshold = 0.3;            

float x1 = 0.479;
float x2 = 0.39;
float x3 = 0.366;
std::mutex mtx_topic;
std::mutex mtx_clu;

// 读取文件内容到内存
void read_data(const std::string& text_file_path, const std::string& embeddings_file_path,
               std::vector<std::vector<std::string>>& text_data,
               std::vector<std::vector<float>>& embeddings_data) {
    std::ifstream text_file(text_file_path);
    std::ifstream embeddings_file(embeddings_file_path);
    assert(text_file);
    assert(embeddings_file);

    std::string line_t, line_e;
    while (getline(text_file, line_t) && getline(embeddings_file, line_e)) {
        std::istringstream iss_t(line_t);
        std::vector<std::string> line2vec{std::istream_iterator<std::string>{iss_t}, std::istream_iterator<std::string>{}};
        text_data.push_back(line2vec);

        std::vector<float> text_embedding;
        std::istringstream iss_e(line_e);
        std::string item;
        while (getline(iss_e, item, ',')) {
            text_embedding.push_back(std::stof(item));
        }
        embeddings_data.push_back(text_embedding);
    }
}

int main(int argc, char* argv[]) {

    int addCount = 0;

    topic_clu.iniDimandThres(D, alpha, test, check);
    topic_index.ini(beta,  test);

    const std::string text_file_path = "dataset/short_texts.txt";
    const std::string embeddings_file_path = "dataset/short_vectors.csv";

    std::vector<std::vector<std::string>> text_data;
    std::vector<std::vector<float>> embeddings_data;

    read_data(text_file_path, embeddings_file_path, text_data, embeddings_data);

    assert(text_data.size() == embeddings_data.size());
    auto begintime = std::chrono::high_resolution_clock::now();

    int countforPUB = 0;
    int countforSUB = 0;
    auto lasttime = std::chrono::high_resolution_clock::now();
    bool flag = 0;
    for (size_t i = 0; i < text_data.size(); ++i) {
        const auto& text_embedding = embeddings_data[i];
        std::vector<std::string>  line2vec = text_data[i];

        if(flag){
            countforPUB++;
            int topic_find = topic_index.find_topic_HNSW(text_embedding, line2vec);
            if (topic_find < 0) {
                topic_clu.online_add(text_embedding, topic_index, line2vec);
            }

            if (countforPUB % 10000 == 0) {
                topic_clu.remake();
            }     
            if(i==1000000){
                break;
            }
        }else{
            countforSUB++;        
            topic_index.find_topK_topic(embeddings_data[i], SUB_id);
            ++SUB_id;
            
            if(countforSUB == 100000){
                flag = 1;
                i = 1;
            }

        }

    }

    topic_clu.go_to_nearest_topic(topic_index , 100);   

    return 0;
}
