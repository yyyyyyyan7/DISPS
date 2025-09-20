#ifndef topic_cpp
#define topic_cpp
#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <set>
#include <fstream>
#include <queue>
#include "compute_distance.h"
#include "compute_cosine.h"
#include <iostream>
#include <algorithm> 
#include <chrono>
#include <mutex>
class TOPIC_index {
private:
    int overlap = 0;
    struct CompareByFirstBig {         
        constexpr bool operator()(const std::pair<float, int>& a,
                                    const std::pair<float, int>& b) const noexcept {
            return a.first < b.first;  
        }
    };
    struct CompareByFirst {      
        constexpr bool operator()(const std::pair<float, int>& a,
                                    const std::pair<float, int>& b) const noexcept {
            return a.first > b.first;  
        }
    };

public:
    
    int dim = 1024;
    int topic_num = 0;                 
    int k = 10;
    int active_PIVOT_num = 0;                               
    int PIVOTE_id = 0;    
    float threshold_topic;                                 
    bool test_f ;                                 
    float alpha = 0.66;
    float beta2 = 0.76;
    float LAMDA = 0.0004;                           
    bool use_single_distance = false;
    std::mutex mtx1;        
    std::mutex mtx2;
    std::mutex mtx3;
    std::mutex mtx31;
    std::mutex mtx32;
    std::mutex mtx33;
    std::vector<int> topic_state;                           
    std::vector<std::map<std::string,int>> words_num;        
    std::map<int,std::vector<float>> embedding;      
    std::map<int,std::vector<std::vector<float>>> embedding_for_text;     
    std::map<int,std::vector<float>> SUB_embedding;  
    std::vector<int> tweets_num;                    
    std::vector<int> low_limit_top;                 
    std::map<int, std::map<std::vector<std::string> , int>> Topic_text;              
    int active_topic_num;                                 
    std::mt19937 gen;
    ////////////////for HNSW
    std::set<int> upper_record;                          
    std::map<int,std::set<int>> layer_up_neighbors;     
    std::map<int,std::set<int>> layer_middle_neighbors; 
    std::map<int,std::set<int>> layer_down_neighbors;   

    std::map<int,int> topic_latest_update;         
    int global_time_index = 0;                  
    std::map<int,float> top_K_limit;                    
    ///////////////for HNSW over
    std::map<int,int> pivot_TOPIC;                                         
    std::map<int,std::set<int>> TOPIC_pivot;                                

    //for SUB
    std::map<int,std::set<int>> SUB_Candidate_topic;                                  
    std::map<int,std::set<int>> SUB_topk_topic;
    std::map<int,std::set<int>> Pivot_SUB;       
    std::map<int,std::set<int>> topic_SUB;              
    std::map<int,std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirstBig>> SUB_TOPIC;    
    
    std::map<int, std::map<int,float>> Pivot_SUB_list; 
    using TimePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
    std::vector<TimePoint> lastUpdateTime;
   
    
    std::vector<std::string> intersection(const std::vector<std::string>& v1, const std::vector<std::string>& v2) {
        std::vector<std::string> result;
        std::vector<std::string> temp_v1 = v1;
        std::vector<std::string> temp_v2 = v2;

        std::sort(temp_v1.begin(), temp_v1.end());
        std::sort(temp_v2.begin(), temp_v2.end());

        std::set_intersection(temp_v1.begin(), temp_v1.end(), temp_v2.begin(), temp_v2.end(),
                            std::back_inserter(result));
        return result;
    }

    std::vector<std::string> difference(const std::vector<std::string>& v1, const std::vector<std::string>& v2) {
        std::vector<std::string> result;
        std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),
                            std::back_inserter(result));
        return result;
    }
    
    

    TOPIC_index() {
        topic_num=0;
        active_topic_num = 0;
        topic_state.resize(0);
        words_num.resize(0);
        gen.seed(3404);
        global_time_index = 0;                  
    }
    // 构造函数
    TOPIC_index(int num_topics) : topic_num(num_topics) {
        topic_state.resize(num_topics);
        words_num.resize(num_topics);
        tweets_num.resize(num_topics);
    }

    void ini(float thd , bool test){ 
        gen.seed(3404);
        topic_num = 0;
        active_topic_num = 0;
        topic_state.resize(0);
        words_num.resize(0);
        tweets_num.resize(0);
        threshold_topic = thd;                  
        words_num.resize(0);
        this->test_f = test;
    }

    void erase_pivot(int chosen){   
        if(test_f) std::cout<<"erase_pivot"<<std::endl;

        active_PIVOT_num--;
        TOPIC_pivot[pivot_TOPIC[chosen]].erase(chosen);
        pivot_TOPIC.erase(chosen);
        embedding.erase(chosen);
        if(upper_record.find(chosen)!=upper_record.end()){
            upper_record.erase(chosen);
            for(int neighbor:layer_up_neighbors[chosen]){
                layer_up_neighbors[neighbor].erase(chosen);
            }
            layer_up_neighbors.erase(chosen);
        }
        if(layer_middle_neighbors.find(chosen)!=layer_middle_neighbors.end()){
            for(int neighbor:layer_middle_neighbors[chosen]){
                layer_middle_neighbors[neighbor].erase(chosen);
            }
            layer_middle_neighbors.erase(chosen);
        }
        for(int neighbor:layer_down_neighbors[chosen]){
            layer_down_neighbors[neighbor].erase(chosen);
        }
        layer_down_neighbors.erase(chosen);

        if(test_f)std::cout<<"erase_pivot over"<<std::endl;
    }

    

    void Generate_pivot(int chosen ){  
        std::vector<float> centroids[2];
        float radius_sum;
        std::uniform_int_distribution<> dis(0, embedding_for_text[chosen].size() - 1);
        int random_offset1 =  dis(gen);
        int random_offset2 =  dis(gen);  
        while(random_offset1 == random_offset2){
            random_offset2 =  dis(gen);
        }
        centroids[0] = embedding_for_text[chosen][random_offset1];   
        centroids[1] = embedding_for_text[chosen][random_offset2];  

        for(int i=0 ; i<3 ; i++){        
            radius_sum = 0;
            std::vector<float> new_centroids[2];
            for(int j=0;j<2;j++)    new_centroids[j] = std::vector<float>(dim, 0);
            
            for(std::vector<float>& text_vec : embedding_for_text[chosen]){
                float dis[2];
                if(text_vec.size() != 1024){
                        continue;
                }
                for(int j=0;j<2;j++){
                     
                    dis[j] = compute_l2_distance(text_vec,centroids[j]);
                }

               
                radius_sum = std::max(dis[0]+dis[1] , radius_sum);
                if(dis[0]<dis[1]){   
                    for(int i=0;i<dim;i++){
                        new_centroids[0][i]+=text_vec[i];
                    }
                }else{    
                    for(int i=0;i<dim;i++){
                        new_centroids[1][i]+=text_vec[i];
                    }
                }
            }
            for(int j=0;j<2;j++){     
                centroids[j] = new_centroids[j];
            }
        
        }

        int new_pivot = PIVOTE_id++; 
        TOPIC_pivot[chosen].clear();
        TOPIC_pivot[chosen].insert(new_pivot);
        embedding[new_pivot] = centroids[0];
        pivot_TOPIC[new_pivot] = chosen;
        Insert_PIVOT_into_HNSW(new_pivot); 
        new_pivot = PIVOTE_id++;
        TOPIC_pivot[chosen].insert(new_pivot);
        embedding[new_pivot] = centroids[1];
        pivot_TOPIC[new_pivot] = chosen;
        Insert_PIVOT_into_HNSW(new_pivot); 
        
    } 

  

    void check_nearby_topic_for_emerge(int chosen){   

        std::set<int> nearby_topic;
        for(int topic_i : TOPIC_pivot[chosen]){
            for(int neighbor:layer_down_neighbors[topic_i]){
                nearby_topic.insert(pivot_TOPIC[neighbor]);
                for(int neighbor2:layer_down_neighbors[neighbor]){
                    nearby_topic.insert(pivot_TOPIC[neighbor2]);
        
                }
            }
        }
        float min_avg_dis = 1000000;
        int min_id = -1;
        std::set<int> merge_set;;
        for(int j : nearby_topic){
            if(j==chosen)continue;
            if(TOPIC_pivot[chosen].size()==0 || TOPIC_pivot[j].size()==0)continue;
                float avg_pivot_dis = 0;
           
                int count = 0;
                for(int p1:TOPIC_pivot[chosen]){
                    for(int p2:TOPIC_pivot[j]){
                        float d = compute_l2_distance(embedding[p1],embedding[p2]);

                        avg_pivot_dis+=d;
                        count++;
                    }
                }
                avg_pivot_dis/=count;
           
                if(avg_pivot_dis < min_avg_dis){
                    min_avg_dis = avg_pivot_dis;
                    min_id = j;
                }
        }
        float merge_thresold = alpha/(std::log(tweets_num[chosen])+1);
        if(min_avg_dis < merge_thresold){
            merge_set.insert(min_id);
        }
        if(merge_set.size()==0)return;
        merge_set.insert(chosen);
        merge_topic(merge_set);   
        

    }

    void update_PIVOT(int chosen){       
        if(test_f)std::cout<<"update_PIVOT"<<std::endl;
        
        if(TOPIC_pivot[chosen].size() == 0 || TOPIC_pivot[chosen].size()>6){          
            if(Topic_text[topic_num].size()<=3){
                return;
            }
            Generate_pivot(chosen);
            
        }else{                                    
            
        }
        check_nearby_topic_for_emerge(chosen);
        if(test_f)std::cout<<"update_PIVOT over"<<std::endl;
    }

    int merge_topic(std::set<int> merge_set){     
        if(test_f)std::cout<<"merge_topic"<<std::endl;
        int chosen = *merge_set.begin();
        merge_set.erase(chosen);
 
        
        for(int x:merge_set){          
            for(int p_id:TOPIC_pivot[x]){  
                TOPIC_pivot[chosen].insert(p_id);
                pivot_TOPIC[p_id] = chosen;
            }
        
            Topic_text[chosen].insert(Topic_text[x].begin(), Topic_text[x].end());
            Topic_text[x].clear();
            
            tweets_num[chosen]+=tweets_num[x];
            tweets_num[x] = 0;
            TOPIC_pivot.erase(x);
            active_topic_num--;
        }
      
        if(test_f)std::cout<<"merge_topic over"<<std::endl;
        return chosen;
    }

    int find_topic_HNSW(std::vector<float> vector, std::vector<std::string>& line2vec){ 
        if(test_f)std::cout<<"find_topic_HNSW"<<std::endl;
        if(active_topic_num==0){return -2;}

        std::uniform_int_distribution<> dis(0, upper_record.size() - 1);
        int random_offset =  dis(gen);

        auto it = upper_record.begin();
        std::advance(it, random_offset);
        int curr_obj = *it;
        int very_start = curr_obj;
        float curr_dist = compute_l2_distance(vector, embedding[curr_obj]);
        std::map<int,std::set<int>> *layer_x_neighbors; 
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> nearest_PIVOT; 
        
        std::set<int> passed;
        for(int level = 3; level>0; level--){
            switch(level){
                case 1:  layer_x_neighbors = &layer_down_neighbors ; break;
                case 2:  layer_x_neighbors = &layer_middle_neighbors ; break;
                case 3:  layer_x_neighbors = &layer_up_neighbors ;
            }
            bool changed = true;
            
            while(changed){
                changed = false;
                std::set<int> tobeocompute;
                for(int nei : (*layer_x_neighbors)[curr_obj]){
                    tobeocompute.insert(nei);
                    for(int nei2 : (*layer_x_neighbors)[nei]){
                        tobeocompute.insert(nei2);
                       
                    }
                }
                for (int cand : tobeocompute) {
                    if(passed.find(cand)!=passed.end())continue;
                    passed.insert(cand);
                    float d = compute_l2_distance(vector, embedding[cand]);
                    nearest_PIVOT.emplace(d,cand);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = cand;
                        changed = true;
                    }
                }
            }
        }
        std::map<int , float> topic_dist_first;
        std::map<int , float> topic_dist_twice;

        int pivot_nearest = nearest_PIVOT.top().second;  
        float pivot_nearest_dist = nearest_PIVOT.top().first;
        while(nearest_PIVOT.size()!=0){
            int pivot = nearest_PIVOT.top().second;        
            float pivot_dist = nearest_PIVOT.top().first;
           // std::cout<<"pivot_dist: "<<pivot_dist;
            nearest_PIVOT.pop();
            if(topic_dist_first.find(pivot_TOPIC[pivot])==topic_dist_first.end()){
                topic_dist_first[pivot_TOPIC[pivot]] = pivot_dist;
            }else if(topic_dist_twice.find(pivot_TOPIC[pivot])==topic_dist_twice.end()){
                topic_dist_twice[pivot_TOPIC[pivot]] = pivot_dist + topic_dist_first[pivot_TOPIC[pivot]];
            }
        }
   
        curr_obj = -1;
        curr_dist = 1000000; 
        for(auto x:topic_dist_twice){
            if(x.second<curr_dist){
                curr_dist = x.second;
                curr_obj = x.first;
            }
        }

        if(test_f)std::cout<<"find_topic_HNSW over"<<std::endl;

        //std::cout<<"curr_dist: "<<curr_dist<<" "<<threshold_topic<<std::endl;
        ///////////////////////////////test
        // curr_dist = pivot_nearest_dist;
        // std::cout<<"curr_dist: "<<curr_dist<<" "<<curr_obj<<std::endl;
        // curr_obj = pivot_nearest;
        ///////////////////////////////test
        //threshold_topic = alpha/(std::log(tweets_num[curr_obj])+1)
        if(use_single_distance){
            //std::cout<<"curr_dist: "<<curr_dist<<"  ";
            curr_dist = pivot_nearest_dist*2;
            curr_obj = pivot_TOPIC[pivot_nearest];
            //std::cout<<"curr_dist: "<<curr_dist<<" "<<threshold_topic<<std::endl;
        }
        if(curr_dist < threshold_topic){                    
            // curr_dist +=0.4;
            // radius[curr_obj] = std::max(radius[curr_obj] , curr_dist);
            addItem(curr_obj , line2vec, vector);
            global_time_index++;
            topic_latest_update[curr_obj] = global_time_index;
            return curr_obj;
        }


        return -1;  

    }

    int addItem(int chosen, std::vector<std::string> line , std::vector<float> vector){  
        embedding_for_text[chosen].push_back(vector);
        if(test_f) std::cout<<"addItem"<<std::endl;
        Topic_text[chosen][line] = 0;    
        tweets_num[chosen]++;
        bool flag = false;
        
        if(test_f)std::cout<<"addItem over"<<std::endl;
        
    
        float min_dis = 1000000;
        int min_id = -1;
        for(int pivot_id:TOPIC_pivot[chosen]){
            float d = compute_l2_distance(embedding[pivot_id],vector);
            if(d<min_dis){
                min_dis = d;
                min_id = pivot_id;
            }
        }
        if(min_dis > threshold_topic/2){
            int new_pivot = PIVOTE_id++;
            TOPIC_pivot[chosen].insert(new_pivot);
            embedding[new_pivot] = vector;
            pivot_TOPIC[new_pivot] = chosen;
            Insert_PIVOT_into_HNSW(new_pivot); 
        }
        return 0;        
    }

    void findNewSUB(int chosen){    
        if(test_f)std::cout<<"findNewSUB"<<std::endl;
        std::set<int> candidates;   
        std::set<int> SUB_candidates;
        std::map<int , std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>> SUB_PIVOT_distances;
    
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> final_SUB_candidates;
        for(int pivot_id:TOPIC_pivot[chosen]){
           
            for(int x:layer_down_neighbors[pivot_id]){
                for(int i:topic_SUB[x]){
                    float d = compute_l2_distance(SUB_embedding[i],embedding[pivot_id]);
                    SUB_PIVOT_distances[i].emplace(d,pivot_id);
                }
            }
        }
        for(auto x:SUB_PIVOT_distances){
            if(x.second.size()>=2){                 
                float d1 = x.second.top().first;
                x.second.pop();
                float d2 = x.second.top().first;
                final_SUB_candidates.emplace(d1+d2 , x.first);
            }
        }
        topic_SUB[chosen].clear(); 
        for(int i=0;i<k && final_SUB_candidates.size()!=0;i++){
            topic_SUB[chosen].insert(final_SUB_candidates.top().second);
            SUB_TOPIC[final_SUB_candidates.top().second].emplace(final_SUB_candidates.top().first,chosen);
            final_SUB_candidates.pop();
            if(SUB_TOPIC[final_SUB_candidates.top().second].size()>k){   
                SUB_TOPIC[final_SUB_candidates.top().second].pop();
            }
            top_K_limit[final_SUB_candidates.top().second] = SUB_TOPIC[final_SUB_candidates.top().second].top().first;  
        }

        if(test_f)std::cout<<"findNewSUB over"<<std::endl;
    }
    
   void up2upper_layer(int chosen , int very_start , std::set<int> middle_nei){   
        if(test_f)std::cout<<"up2upper_layer"<<std::endl;
        if(upper_record.find(chosen)!=upper_record.end()){
            return ; 
        }
        layer_up_neighbors[chosen] = std::set<int>();  
        layer_up_neighbors[chosen].insert(very_start);  
        layer_up_neighbors[very_start].insert(chosen);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;

        for (int i : upper_record) {
            
            float tmpdis = compute_l2_distance(embedding[chosen], embedding[i]);
            pq.push({tmpdis, i});
        }
        int n = 5; 
        while (n-- && !pq.empty()) {
            auto top = pq.top();
            pq.pop();
            layer_up_neighbors[chosen].insert(top.second);
            layer_up_neighbors[top.second].insert(chosen);
        }
        upper_record.insert(chosen);
        up2middlelayer(chosen,middle_nei);
        if(test_f)std::cout<<"up2upper_layer over"<<std::endl;
    }

    void up2middlelayer(int chosen , std::set<int> middle_nei){   
        if(test_f)std::cout<<"up2middlelayer"<<std::endl;
        std::set<int> search_range;
        layer_middle_neighbors[chosen] = std::set<int>();  
        for(int x: middle_nei){
            search_range.insert(layer_middle_neighbors[x].begin(),layer_middle_neighbors[x].end());
            layer_middle_neighbors[chosen].insert(x);  
            layer_middle_neighbors[x].insert(chosen);
        }
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> pq;

    
        for (int i : search_range) {                        
            
            float tmpdis = compute_l2_distance(embedding[chosen], embedding[i]);
            
            pq.emplace(tmpdis, i);
        }

       
        int n = 20; 
        while (n-- && !pq.empty()) {
            auto top = pq.top();
            pq.pop();
            layer_middle_neighbors[chosen].insert(top.second);
            layer_middle_neighbors[top.second].insert(chosen);
        }
        upper_record.insert(chosen);
        if(test_f)std::cout<<"up2middlelayer over"<<std::endl;
    }
    
    void GTopic(std::vector<std::string> line , std::vector<float> vector){  
        if(test_f){
            std::cout<<"GTopic"<<std::endl;
            std::ofstream outfile("LOG.txt",std::ios::app);
            outfile<<"born into "<<topic_num<<" by ";
            for(const std::string& word:line){
                outfile<<word<<" ";
            }
            outfile<<std::endl;
            outfile.close();
        }
        embedding_for_text[topic_num].push_back(vector);
        tweets_num[topic_num]++;
        Topic_text[topic_num][line] = 1 ;    
      

        if(test_f)std::cout<<"GTopic over"<<std::endl;
        return ;

    }

    void begin_add_topic(){  
        addTopic();
    }

    void end_add_topic(float r ,bool flg){      
        if(test_f)std::cout<<"end_add_topic"<<std::endl;
        
        topic_SUB[topic_num] = std::set<int>();    
        if(flg)update_PIVOT(topic_num);  
        topic_latest_update[topic_num] = global_time_index;

        topic_num++;  
        if(flg)check_old_topic();
        
       if(test_f) std::cout<<"end_add_topic over"<<std::endl;
    }

    void check_old_topic(){
        if(test_f)std::cout<<"check_old_topic"<<std::endl;
        int thres_topic = global_time_index - 100000;
        if(thres_topic<=0)return ;       
        std::set<int> to_be_erase;
        for(auto x : topic_latest_update){
            if(x.second < thres_topic){         
                erase_old_topic(x.first);
                to_be_erase.insert(x.first);
            }
        }
        for(int i:to_be_erase){
            topic_latest_update.erase(i);
        }
        if(test_f)std::cout<<"check_old_topic over"<<std::endl;
    }
    void reshape_top_k(int sub_id ,int chosen){  
        if(test_f)std::cout<<"reshape_top_k"<<std::endl;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirstBig> new_topk;
        std::set<int> cand;
        SUB_topk_topic[sub_id].erase(chosen);
        
        while (!SUB_TOPIC[sub_id].empty()) {
            auto pair = SUB_TOPIC[sub_id].top();
            SUB_TOPIC[sub_id].pop();
            if (pair.second != chosen) {
                new_topk.push(pair);   
            }
        }
        for(int i : SUB_Candidate_topic[sub_id]){
            if(SUB_topk_topic[sub_id].find(i) == SUB_topk_topic[sub_id].end()){
                cand.insert(i);
            }
        }
        float min_dis = 1000000;
        int min_id = -1;
        int min_pid1 = -1;
        int min_pid2 = -1;
        for(int i : cand){
            float dist_to_topic;
            float min_dis = 1000000;
            float min_dis2 = 1000000;
            int min_id = -1;
            int min_id2 = -1;
            for(int p:TOPIC_pivot[i]){
                if(Pivot_SUB_list[p].find(sub_id)==Pivot_SUB_list[p].end())continue;
                float tmp = Pivot_SUB_list[p][sub_id];
                if(tmp<min_dis){
                    min_dis2 = min_dis;
                    min_id2 = min_id;
                    min_dis = tmp;
                    min_id = p;
                }else if(tmp<min_dis2){
                    min_dis2 = tmp;
                    min_id2 = p;
                }
            }
            dist_to_topic = min_dis + min_dis2;  
            if(min_dis > dist_to_topic){
                min_dis = dist_to_topic;
                min_id = i;
                min_pid1 = min_id;
                min_pid2 = min_id2;
            }
        }
        new_topk.push(std::make_pair(min_dis,min_id));
        topic_SUB[min_id].insert(sub_id);
        Pivot_SUB[min_pid1].insert(sub_id);
        Pivot_SUB[min_pid2].insert(sub_id);
        SUB_topk_topic[sub_id].insert(pivot_TOPIC[min_id]);

        SUB_TOPIC[sub_id] = new_topk;
        top_K_limit[sub_id] = SUB_TOPIC[sub_id].top().first;
        if(test_f)std::cout<<"reshape_top_k over"<<std::endl;
    }

    void erase_old_topic(int chosen){  
;
        for(int pi:TOPIC_pivot[chosen]){
            erase_pivot(pi);
        }
        TOPIC_pivot.erase(chosen);
        Topic_text[chosen].clear();
        tweets_num[chosen] = 0;
        active_topic_num--;
        
        for(int sub : topic_SUB[chosen]){
            SUB_Candidate_topic[sub].erase(chosen);
        }

        for(int sub : topic_SUB[chosen]){
            reshape_top_k(sub , chosen); 
        }

        topic_SUB[chosen].clear();
        topic_state[chosen] = 4;
    } 
    void Insert_PIVOT_into_HNSW(int pivot_id){
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW: "<<active_PIVOT_num<<std::endl;
        int p_id = pivot_id;
        active_PIVOT_num ++;    

        if(active_PIVOT_num==1){    
            layer_up_neighbors[pivot_id] = std::set<int>();     
            layer_down_neighbors[pivot_id] = std::set<int>();  
            upper_record.insert(pivot_id);
            layer_middle_neighbors[pivot_id] = std::set<int>(); 
            return;
        }
    
        std::uniform_int_distribution<> dis(0, upper_record.size() - 1);
        int random_offset =  dis(gen);

        auto it = upper_record.begin();
        std::advance(it, random_offset);
        int curr_obj = *it;
        int very_start = curr_obj;
        float curr_dist = 10000;
        float lay_high_dist = 10000;
        std::map<int,std::set<int>> *layer_x_neighbors; 
        std::set<int> middle_nei;
        int middle_nearest;
        int upper_nearest;
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW123: "<<active_PIVOT_num<<std::endl;
        std::set<int> passed;
        for(int level = 3; level>0; level--){           
            switch(level){
                case 1:  layer_x_neighbors = &layer_down_neighbors ; break;
                case 2:  layer_x_neighbors = &layer_middle_neighbors ; break;
                case 3:  layer_x_neighbors = &layer_up_neighbors ; 
            }
            bool changed = true;
            while(changed){
                changed = false;
                std::set<int> candidates;
                candidates.insert(curr_obj);                    
                for (int cand : (*layer_x_neighbors)[curr_obj]) {
                    candidates.insert(cand);
                    for (int cand2 : (*layer_x_neighbors)[cand]) {
                        candidates.insert(cand2);
                    }
                }
                for (int cand : candidates) {
                    if(passed.find(cand)!=passed.end())continue;
                    passed.insert(cand);
                    float d = compute_l2_distance(embedding[pivot_id], embedding[cand]);
                    std::uniform_real_distribution<float> dis1(0.0, 10.0);
                    float rd = dis1(gen);
                    if(level>=2){
                        if(rd < 3.0){                          
                            middle_nei.insert(cand);
                        }
                    }
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = cand;
                        changed = true;
                        if(level>=2) middle_nei.insert(cand);
                    }
                }
            }
            if(level == 3){
                lay_high_dist = curr_dist;
                upper_nearest = curr_obj;
            }
            if(level == 2){
                middle_nearest = curr_obj;
            }

            if(level == 1){  
                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> down_nei;
                std::set<int> tobeadd;
                for(int nei : layer_down_neighbors[curr_obj]){
                    tobeadd.insert(nei);
                    for(int nei2 : layer_down_neighbors[nei]){
                        tobeadd.insert(nei2);
                       
                    }
                }
                for(int nei : tobeadd){
                    float d = compute_l2_distance(embedding[pivot_id], embedding[nei]);
                    down_nei.emplace(d , nei);
                }
                for(int i=0 ; i<k ; i++){           
                    if(down_nei.size()==0)break;
                    int cand = down_nei.top().second;
                    down_nei.pop();
                    layer_down_neighbors[pivot_id].insert(cand);
                    layer_down_neighbors[cand].insert(pivot_id);

                }
                
                layer_down_neighbors[middle_nearest].insert(pivot_id);  
                layer_down_neighbors[pivot_id].insert(middle_nearest);

            }
                
        }
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW level ok  "<<std::endl;
        if((1.0*(float)upper_record.size()/active_PIVOT_num)<0.30){
            if(upper_record.size()>10 && lay_high_dist<0.4){    
                
            }else{
                std::uniform_real_distribution<float> dis(0.0, 10.0);
                float random_offset = dis(gen);
                if(random_offset < lay_high_dist){ 
                    up2upper_layer(pivot_id , very_start , middle_nei);
                }else if(random_offset < 2.0){ 
                    up2middlelayer(pivot_id , middle_nei);
                }
            }
        }
        std::set<int> candidate_sub;
        for(int nei: layer_down_neighbors[pivot_id]){
            candidate_sub.insert(Pivot_SUB[nei].begin(),Pivot_SUB[nei].end());
        }
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW level ok3  "<<std::endl;
        for(int sub_id:candidate_sub){
            if(SUB_embedding.find(sub_id)==SUB_embedding.end()){
                std::cout<<"error:  sub_id not found"<<std::endl;
            }
            if(embedding.find(pivot_id)==embedding.end()){
                std::cout<<"error:  pivot_id not found"<<std::endl;
            }
            float dist = compute_l2_distance(SUB_embedding[sub_id],embedding[pivot_id]);
            Pivot_SUB_list[pivot_id][sub_id] = dist;
        }
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW level ok2  "<<std::endl;
        for(int sub_id:candidate_sub){
            if(SUB_topk_topic[sub_id].find(pivot_TOPIC[pivot_id])==SUB_topk_topic[sub_id].end()){ 
                float dist_to_topic;
                float min_dis = 1000000;
                float min_dis2 = 1000000;
                int min_id = -1;
                int min_id2 = -1;
                for(int p:TOPIC_pivot[pivot_TOPIC[pivot_id]]){
                    if(Pivot_SUB_list[p].find(sub_id)==Pivot_SUB_list[p].end())continue;
                    float tmp = Pivot_SUB_list[p][sub_id];
                    if(tmp<min_dis){
                        min_dis2 = min_dis;
                        min_id2 = min_id;
                        min_dis = tmp;
                        min_id = p;
                    }else if(tmp<min_dis2){
                        min_dis2 = tmp;
                        min_id2 = p;
                    }
                }
                dist_to_topic = min_dis + min_dis2;
                if(SUB_topk_topic[sub_id].size()<k){    

                    Pivot_SUB[min_id].insert(sub_id);
                    Pivot_SUB[min_id2].insert(sub_id);
                    topic_SUB[pivot_TOPIC[min_id]].insert(sub_id);
                    SUB_topk_topic[sub_id].insert(pivot_TOPIC[pivot_id]);
                    SUB_Candidate_topic[sub_id].insert(pivot_TOPIC[pivot_id]);
                    SUB_TOPIC[sub_id].emplace(dist_to_topic,pivot_TOPIC[pivot_id]);
                    if(SUB_topk_topic[sub_id].size()>=k){
                        top_K_limit[sub_id] = SUB_TOPIC[sub_id].top().first;
                    }

                }else{          
                    if(dist_to_topic < top_K_limit[sub_id]){
                        Pivot_SUB[min_id].insert(sub_id);
                        Pivot_SUB[min_id2].insert(sub_id);
                        topic_SUB[pivot_TOPIC[min_id]].insert(sub_id);
                        SUB_topk_topic[sub_id].insert(pivot_TOPIC[pivot_id]);
                        SUB_Candidate_topic[sub_id].insert(pivot_TOPIC[pivot_id]);
                        SUB_TOPIC[sub_id].emplace(dist_to_topic,pivot_TOPIC[pivot_id]);
                        int top_topic = SUB_TOPIC[sub_id].top().second;
                        SUB_topk_topic[sub_id].erase(top_topic);
                        topic_SUB[top_topic].erase(sub_id);
                        for(int p : TOPIC_pivot[top_topic]){
                            Pivot_SUB[p].erase(sub_id);
                        }
                        SUB_TOPIC[sub_id].pop();
                        top_K_limit[sub_id] = SUB_TOPIC[sub_id].top().first;
                    }
                        
                }
            }
        }
        
        if(test_f)std::cout<<"Insert_PIVOT_into_HNSW over"<<std::endl;
    }

    void print_Topic(const char*  file_name){
        std::ofstream ofx(file_name);
        for(int i=0;i<topic_num;i++){
            ofx<<"TOPIC :    "<<i<<"  size:  "<<Topic_text[i].size()<<std::endl;
            for(auto x :Topic_text[i]){
                ofx<<x.second<<" : ";
                for(auto j : x.first){
                    ofx<<j<<" ";
                }
                ofx<<std::endl;
            }
        }
        ofx.close();
    }

    void print_Topic_pivot(const char*  file_name){            
        std::ofstream ofx(file_name);
        for(int i=0;i<topic_num;i++){
            for(int j = i+1;j<topic_num;j++){
                if(TOPIC_pivot[i].size()==0 || TOPIC_pivot[j].size()==0)continue;
                float avg_pivot_dis = 0;
                int count = 0;
                for(int p1:TOPIC_pivot[i]){
                    for(int p2:TOPIC_pivot[j]){
                        float d = compute_l2_distance(embedding[p1],embedding[p2]);
                        avg_pivot_dis+=d;
                        count++;
                    }
                }
                avg_pivot_dis/=count;
                ofx<<"TOPIC: "<<i<<"  TOPIC: "<<j<<"  avg_dis:  "<<avg_pivot_dis<<std::endl;
            }
        }
        for(int i=0;i<topic_num;i++){
            ofx<<"TOPIC: "<<i<<"  pivot num: "<<TOPIC_pivot[i].size()<<std::endl;

        }
        ofx.close();
    }

    void print_Pivot(const char*  file_name){  
        std::ofstream ofx(file_name,std::ios::app);
        for(int i=0;i<topic_num;i++){
            for(int j:TOPIC_pivot[i]){
                for(int k:TOPIC_pivot[i]){
                    ofx<<"TOPIC: "<<i<<"  pivot: "<<j<<"  pivot: "<<k<<"  dis:  "<<compute_l2_distance(embedding[j],embedding[k])<<std::endl;
                }
            }
        }
        ofx.close();
    }
   
    void addTopic(){
        if(test_f)std::cout<<"addtopic"<<std::endl;
        
        active_topic_num++;
        
        topic_state.emplace_back(0);
        
        words_num.emplace_back(std::map<std::string, int>());
        embedding_for_text[topic_num] = std::vector<std::vector<float>>();
        // 
        tweets_num.emplace_back(0);
        // trend.emplace_back(0);
        low_limit_top.emplace_back(0);     //
        TimePoint currentTime = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now());
        // 
        lastUpdateTime.push_back(currentTime);
        if(test_f)std::cout<<"addtopic over"<<std::endl;

    }
    
    void find_topK_topic(std::vector<float>& vector, int SUB_ID) { 
        if(test_f) std::cout << "find_topK_topic" << std::endl;
        SUB_embedding[SUB_ID] = vector;
        if (active_topic_num == 0) return;

        std::uniform_int_distribution<> dis(0, upper_record.size() - 1);
        int random_offset = dis(gen);
        auto it = upper_record.begin();
        std::advance(it, random_offset);
        int curr_obj = *it;
        float curr_dist = 10000;

        // 
        std::unordered_set<int> passed;
        std::unordered_set<int> tobecomputed;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> nearest_PIVOT;
        std::mutex* layer_mutex;
        for (int level = 3; level > 0; level--) {
            
            std::map<int, std::set<int>>* layer_x_neighbors;
            switch(level) {
                case 1: layer_x_neighbors = &layer_down_neighbors; layer_mutex = &mtx1; break;
                case 2: layer_x_neighbors = &layer_middle_neighbors; layer_mutex = &mtx2;break;
                case 3: layer_x_neighbors = &layer_up_neighbors;layer_mutex = &mtx3;break;
            }

            bool changed = true;
            while (changed) {
                std::lock_guard<std::mutex> lock(*layer_mutex);
                changed = false;
                tobecomputed.clear();
                for (int x : (*layer_x_neighbors)[curr_obj]) {
                    tobecomputed.insert(x);

                }

                for (int cand : tobecomputed) {
                    if (passed.find(cand) != passed.end()) continue;
                    passed.insert(cand);

                    float d = compute_l2_distance(vector, embedding[cand]);
                    nearest_PIVOT.emplace(d, cand);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = cand;
                        changed = true;
                    }
                }
            }
        }

        tobecomputed.clear();
        tobecomputed.insert(curr_obj);

        while (nearest_PIVOT.size() < 2 * k && !tobecomputed.empty()) {
            std::unordered_set<int> tmp;
            for (int candi : tobecomputed) {    
                if (passed.find(candi) != passed.end()) continue;
                passed.insert(candi);
                
                float d = compute_l2_distance(vector, embedding[candi]);
                nearest_PIVOT.emplace(d, candi);
            }
            tobecomputed.swap(tmp);
        }

        std::unordered_map<int, float> topic_dist_first, topic_dist_twice;
        std::unordered_map<int, std::unordered_set<int>> topic_pi;

        while (!nearest_PIVOT.empty()) {
            int pivot = nearest_PIVOT.top().second;
            float pivot_dist = nearest_PIVOT.top().first;
            Pivot_SUB_list[pivot][SUB_ID] = pivot_dist;
            nearest_PIVOT.pop();

            int p_topic = pivot_TOPIC[pivot];
            SUB_Candidate_topic[SUB_ID].insert(p_topic);

            if (topic_dist_first.find(p_topic) == topic_dist_first.end()) {
                topic_dist_first[p_topic] = pivot_dist;
                topic_pi[p_topic].insert(pivot);
            } else if (topic_dist_twice.find(p_topic) == topic_dist_twice.end()) {
                topic_dist_twice[p_topic] = pivot_dist + topic_dist_first[p_topic];
                topic_pi[p_topic].insert(pivot);
            }
        }

        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> nearest_TOPIC;
        for (const auto& x : topic_dist_twice) {
            nearest_TOPIC.emplace(x.second, x.first);
        }

        int count = 0;

        while (!nearest_TOPIC.empty() && count < k) {
            //std::lock_guard<std::mutex> lock(mtx);
            count++;
            
            int top_topic = nearest_TOPIC.top().second;
            
            topic_SUB[top_topic].insert(SUB_ID);
            for (int pivot : topic_pi[top_topic]) {
                Pivot_SUB[pivot].insert(SUB_ID);
            }
            SUB_TOPIC[SUB_ID].emplace(nearest_TOPIC.top().first, top_topic);
            SUB_topk_topic[SUB_ID].insert(top_topic);
            nearest_TOPIC.pop();
        }
          
        top_K_limit[SUB_ID] = SUB_TOPIC[SUB_ID].top().first;
        
        if (SUB_Candidate_topic[SUB_ID].size() < k) {
            top_K_limit[SUB_ID] = 1000000;
        }
        
        if(test_f) std::cout << "find_topK_topic over" << std::endl;
    }

    void set_topic_threshold(float new_threshold){
        threshold_topic = new_threshold;
    }
    void update_topic_brutely(int topic){
        for(auto subembedding : SUB_embedding){
            float min_dis1 = 1000000;
            float min_dis2 = 1000000;
            int min_id1 = -1;
            int min_id2 = -1;
            for(int pivot_id : TOPIC_pivot[topic]){
                float dist = compute_l2_distance(subembedding.second , embedding[pivot_id]);
                if(dist<min_dis1){
                    min_dis2 = min_dis1;
                    min_id2 = min_id1;
                    min_dis1 = dist;
                    min_id1 = pivot_id;
                }else if(dist<min_dis2){
                    min_dis2 = dist;
                    min_id2 = pivot_id;
                }
            }
            float dist_to_topic = min_dis1 + min_dis2;
            if(SUB_Candidate_topic[subembedding.first].find(topic)==SUB_Candidate_topic[subembedding.first].end()){
                if(SUB_topk_topic[subembedding.first].size()<k){
                    SUB_topk_topic[subembedding.first].insert(topic);
                    SUB_Candidate_topic[subembedding.first].insert(topic);
                    SUB_TOPIC[subembedding.first].emplace(dist_to_topic,topic);
                    if(SUB_topk_topic[subembedding.first].size()>=k){
                        top_K_limit[subembedding.first] = SUB_TOPIC[subembedding.first].top().first;
                    }
                }else{
                    if(dist_to_topic<top_K_limit[subembedding.first]){
                        int top_topic = SUB_TOPIC[subembedding.first].top().second;
                        SUB_topk_topic[subembedding.first].erase(top_topic);
                        SUB_Candidate_topic[subembedding.first].erase(top_topic);
                        SUB_TOPIC[subembedding.first].pop();
                        SUB_topk_topic[subembedding.first].insert(topic);
                        SUB_Candidate_topic[subembedding.first].insert(topic);
                        SUB_TOPIC[subembedding.first].emplace(dist_to_topic,topic);
                        top_K_limit[subembedding.first] = SUB_TOPIC[subembedding.first].top().first;
                    }
                }
            }
        }
    }
    

    float compute_l2_distance(std::vector<float> x, std::vector<float> y) {
        size_t d = dim;
        const float* xArray = x.data();
        const float* yArray = y.data();
        float result;
        result = CosineDistanceAVX(xArray, yArray, &d);            
        return result;
    }

};
#endif



