
#ifndef ONLINE_CLUSTER_H1
#define ONLINE_CLUSTER_H1
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <map>
#include <set>
#include <queue>
#include <chrono>
#include <thread>
#include <cmath>
#include <random>
#include <unordered_set>
#include <omp.h>
#include <bitset>
#include "topic.cpp"
#include "compute_distance.h"
#include "compute_cosine.h"
#include <numeric>


class TOPIC_clu{                     
    public:
    int global_index = 0;

    int dim = 1024;      
    
    int outdatenum=0;            
    
    float alpha = 0.66;
   
    float max_radius;
    float topic_lim;
    bool test_f ;
    int check_lim = 3;
    
    std::map<int,std::vector<std::string>> hidden_layer_line;

    std::map<int,std::set<int>> upper_layer_neighbors; 

    std::map<int,std::set<int>> middle_layer_neighbors; 

    std::map<int,std::vector<float>>middle_center_vectors;

    std::map<int,std::set<int>> hidden_clu_members;  

    std::map<int,std::set<int>> small_neibors; 

    std::map<int, int> hidden_clus;         

    std::map<int,float> radius;            


    std::set<int> upper_hidden_record; 

    std::map<int,int> id2topic;
    std::mt19937 gen;
    std::map<int,std::vector<float>>hidden_vectors;

    struct CompareByFirst {
        constexpr bool operator()(const std::pair<float, int>& a,
                                    const std::pair<float, int>& b) const noexcept {
            return a.first > b.first;  
        }
    };
    int oodT = 0;    
    void iniDimandThres(int x , float alpha1,  bool test ,int check){                            
        this->check_lim = check;
        this->dim = x;
        this->alpha = alpha1;
        //this->online_threshold = r;
        this->max_radius =  alpha  ;  
        this->topic_lim = 1-cos(max_radius);
        this->test_f = test;
        gen.seed(3407);
    }
    float x1,x2,x3;
    void iniDimandThres(int x , float x1, float x2 , float x3 , bool test ,int check){                            
        this->check_lim = check;
        this->dim = x;
        this->x1 = x1;
        this->x2 = x2;
        this->x3 = x3;
   
        this->max_radius =  alpha/(std::log(2)+1);                
        this->topic_lim = 1-cos(max_radius);
        this->test_f = test;
        gen.seed(3407);
    }


    void print_all(){
        std::ofstream off("record.txt",std::ios::app);
        off<<" ///////////////////////////////////////////// "<<std::endl;
        for(auto i : hidden_clu_members){
            off<<"id:   "<<i.first<<"  r: "<<radius[i.first]<<"  mems num: "<<hidden_clu_members[i.first].size()<<std::endl;
        }
        off.close();
    }

    bool checkTopic(int checknum){                          
        
        if(hidden_clu_members[checknum].size()> check_lim ){             
            return true;
        }

        return false;
    }

    void complex_update(int group_id ,int pre_clu_id){
        if(test_f)
         std::cout<<"complex_update"<<std::endl;
        float r_min = 0;
        std::vector<float> best_res = middle_center_vectors[pre_clu_id];
        int test_times = 5;

        for(int i = 0 ; i<test_times ;i++){
            std::uniform_int_distribution<> dis(0, hidden_clu_members[pre_clu_id].size() - 1);
            int random_offset =  dis(gen);
            auto it = hidden_clu_members[pre_clu_id].begin();
            std::advance(it, random_offset);
            int start = *it;
            int now = start;
            float tmp_max = -1;
            bool flag = true;
            std::set<int> passed ;
            while(flag){
                flag = false;
                int min_one = -1;

                std::set<int> tobecompute;
                for(int nei1 : small_neibors[now]){
                    tobecompute.insert(nei1);
                    for(int nei2 : small_neibors[nei1]){
                        tobecompute.insert(nei2);
                    }
                }
                for(int cand : tobecompute){
                    if(passed.find(cand)!=passed.end())continue;
                    passed.insert(cand);
                    float dis = compute_l2_distance(hidden_vectors[start],hidden_vectors[cand]);
                    if(dis > tmp_max){
                        tmp_max = dis;
                        min_one = cand;
                        flag = true;
                    }
                }
                if(min_one!=-1)now = min_one;
            } 
           
            start = now;
            tmp_max = -1;
            flag = true;
            passed.clear();
            while(flag){
                flag = false;
                int min_one = -1;

                std::set<int> tobecompute;
                for(int nei1 : small_neibors[now]){
                    tobecompute.insert(nei1);
                    for(int nei2 : small_neibors[nei1]){
                        tobecompute.insert(nei2);
                    }
                }
                for(int cand : tobecompute){
                    if(passed.find(cand)!=passed.end())continue;
                    passed.insert(cand);
                    float dis = compute_l2_distance(hidden_vectors[start],hidden_vectors[cand]);
                    if(dis>tmp_max){
                        tmp_max = dis;
                        min_one = cand;
                        flag = true;
                    }
                }
                if(min_one!=-1)now = min_one;
            }
            float r_tmp = 1.0-cos(acos(1.0-compute_l2_distance(hidden_vectors[start],hidden_vectors[now]))/2);
            if(r_tmp>r_min){
                r_min = r_tmp;

                for(int x = 0;x<hidden_vectors[start].size();x++){
                    best_res[x] = (hidden_vectors[start][x] + hidden_vectors[now][x])/2;
                }
            }
            
        }
        radius[group_id] = r_min;
        middle_center_vectors[group_id] = best_res;
    }

    void justEraseOne(int chosen){    
      
        hidden_clu_members[hidden_clus[chosen]].erase(chosen);        
        hidden_clus.erase(chosen);
        radius.erase(chosen);
        middle_center_vectors.erase(chosen);
        for(int x : middle_layer_neighbors[chosen]){
            middle_layer_neighbors[x].erase(chosen);
        }
        for(int x: small_neibors[chosen]){
            small_neibors[x].erase(chosen);
        }
        middle_layer_neighbors.erase(chosen);

        if(upper_hidden_record.find(chosen) != upper_hidden_record.end()){
            for(int x : upper_layer_neighbors[chosen]){
                upper_layer_neighbors[x].erase(chosen);
            }
            upper_layer_neighbors.erase(chosen); 
        }
        upper_hidden_record.erase(chosen);
        hidden_layer_line.erase(chosen);
      
    }

    void eraseOne(int chosen){                      
       
        middle_center_vectors.erase(chosen);
        for(int y : middle_layer_neighbors[chosen]){
            middle_layer_neighbors[y].erase(chosen);
        }
        for(int y : upper_layer_neighbors[chosen]){
            upper_layer_neighbors[y].erase(chosen);
        }
        for(int x : hidden_clu_members[chosen]){
            hidden_vectors.erase(x);
            // ofx<<chosen<<" erase2:  "<<x<<" left size: "<<down_layer_neighbors[chosen].size()<<std::endl;
            //tags.erase(x);
            hidden_clus.erase(x);
            hidden_layer_line.erase(x);
            middle_layer_neighbors.erase(x);
            upper_layer_neighbors.erase(x);
            upper_hidden_record.erase(x);  
            small_neibors.erase(x);
        }
        // ofx.close();
        // for(int x : middle_layer_neighbors[chosen]){
        //     middle_layer_neighbors[x].erase(chosen);
        // }
        hidden_vectors.erase(chosen);
        hidden_clu_members.erase(chosen);

        radius.erase(chosen);

    }

    void up2upper_layer(int chosen , int very_start){  
        if(upper_hidden_record.find(chosen)!=upper_hidden_record.end()){
            return ; 
        }
        upper_layer_neighbors[chosen] = std::set<int>();  
        upper_layer_neighbors[chosen].insert(very_start);
        upper_layer_neighbors[very_start].insert(chosen);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;

        
        for (int i : upper_hidden_record) {
            
            float tmpdis = compute_l2_distance(middle_center_vectors[chosen], middle_center_vectors[i]);
            
            pq.push({tmpdis, i});
        }

     
        int n = 10; 
      
        while (n-- && !pq.empty()) {
            auto top = pq.top();
            pq.pop();
            upper_layer_neighbors[chosen].insert(top.second);
            upper_layer_neighbors[top.second].insert(chosen);
        }
        upper_hidden_record.insert(chosen);
    }

    void generateTopic(int chosen,TOPIC_index& topic_index){    
        if(test_f)std::cout<<"generateTopic "<<std::endl;

        id2topic[chosen] = topic_index.topic_num;
        
        topic_index.begin_add_topic();  
        for(int x : hidden_clu_members[chosen]){ //
            topic_index.GTopic(hidden_layer_line[x] , hidden_vectors[x]);
            
        }       
        topic_index.GTopic(hidden_layer_line[chosen],hidden_vectors[chosen]);
        topic_index.end_add_topic(radius[chosen], true);    
        
        eraseOne(chosen);

    }
    

    

    void online_add(std::vector<float> vector,TOPIC_index& topic_index , std::vector<std::string> line2vec){
            if(test_f)std::cout<<"start: "<<upper_hidden_record.size()<<std::endl;

                                                  
            hidden_layer_line[++global_index] = line2vec;
            hidden_vectors[global_index] = vector;
            //down_layer_neighbors[global_index] = std::set<int>();  
            hidden_clus[global_index] = global_index;  
            middle_layer_neighbors[global_index] = std::set<int>();
            radius[global_index] =   alpha/(std::log(2)+1) ;                

            if(upper_hidden_record.size()==0){              
                upper_layer_neighbors[global_index] = std::set<int>();  
                middle_center_vectors[global_index] = vector;
                upper_hidden_record.insert(global_index);
                
                hidden_clus[global_index] = global_index;
                hidden_clu_members[global_index] =  std::set<int>();
                hidden_clu_members[global_index].insert(global_index);
                

                return ;
            }
            
            if(test_f)std::cout<<"start: "<<std::endl;  

            
            std::uniform_int_distribution<> dis(0, upper_hidden_record.size() - 1);
            int random_offset =  dis(gen);
            if(random_offset>=upper_hidden_record.size())random_offset = upper_hidden_record.size()-1;
            auto it = upper_hidden_record.begin();
            if(test_f)std::cout<<"offset: "<<random_offset<<std::endl;  
            std::advance(it, random_offset);
            int start = *it;
            if(test_f)std::cout<<"start: "<<start<<std::endl; 
            int very_start = start;      
            if(test_f)std::cout<<"start: "<<start<<" ; "<<upper_hidden_record.size()<<" "<<middle_center_vectors[start].size()<<std::endl;

            float minDIS = compute_l2_distance(vector , middle_center_vectors[start]);

            if(test_f)std::cout<<"minDIS123: "<<minDIS<<" ; "<<upper_hidden_record.size()<<std::endl;
          
            while(true){
                int minNUM = -1;
                for(int choice : upper_layer_neighbors[start]){
                    
                    float tmpDIS = compute_l2_distance(vector , middle_center_vectors[choice]);

                    if(tmpDIS<minDIS){
                        minNUM = choice;
                        minDIS = tmpDIS;
                    }
                }
                if(minNUM == -1){
                    break;
                }else{
                    start = minNUM;
                }
            }
            std::set<int> middle_nei_tobeadd = std::set<int>();     
            middle_nei_tobeadd.insert(start);
            int upper_nearest = start;            
            float upper_min_dis = minDIS;                             
            std::set<int> candidate_Set ; 

            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> top_candidates;
            top_candidates.emplace(minDIS, start);
            candidate_Set.insert(start);
            std::set<int> tmp_candidate_set;  
            std::set<int> passed_set =  std::set<int>();  
            passed_set.insert(global_index);
            while(true){                    
                int minNUM = -1;

                
                std::set<int> tobe_set =  std::set<int>();
                for(int nei1 : middle_layer_neighbors[start]){
                    tobe_set.insert(nei1);
                    for(int nei2 : middle_layer_neighbors[nei1]){
                        tobe_set.insert(nei2);
                    }
                }
                for(int choice : tobe_set){
                    if(passed_set.find(choice)!=passed_set.end()){  
                        continue;
                    }else{
                        passed_set.insert(choice);
                    }
                    float tmpDIS = compute_l2_distance(vector , middle_center_vectors[choice]);               
                    if(tmpDIS<minDIS){
                        minNUM = choice;
                        minDIS = tmpDIS;
                    }
                    if(tmpDIS<topic_lim){                                   
                        candidate_Set.insert(choice);
                        top_candidates.emplace(tmpDIS, choice);     
                    }
                    if(tmpDIS < radius[choice]){                  
                        tmp_candidate_set.insert(choice);
                    }
                    std::uniform_real_distribution<> dis3(0.2, 0.8);
                    double random_value = dis3(gen);
                    if(random_value < tmpDIS){
                        middle_nei_tobeadd.insert(choice);         
                    }  
                }

                if(minNUM == -1){
                    break;
                }else{
                    start = minNUM;
                }
            }       
            int countfornei = 0;
                              
            
            passed_set.insert(global_index);
            if(top_candidates.top().first < radius[top_candidates.top().second]){                  
                tmp_candidate_set.insert(top_candidates.top().second);
            }
            while(top_candidates.size()>0){                      
                int choice = top_candidates.top().second;
                top_candidates.pop();
                
                for(int nbs : middle_layer_neighbors[choice]){
                      
                    if(passed_set.find(nbs)!=passed_set.end()){  
                        continue;
                    }else{
                        passed_set.insert(nbs);
                    }

                    countfornei++;
                    float tmpDIS = compute_l2_distance(vector , middle_center_vectors[nbs]);
                    if(tmpDIS<topic_lim){
                        top_candidates.emplace(tmpDIS,nbs);   
                    }
                    if(tmpDIS<radius[choice]){                  
                        tmp_candidate_set.insert(nbs);
                    }
                    std::uniform_real_distribution<> dis3(0.15, 0.85);
                    double random_value = dis3(gen);
                    if(random_value < tmpDIS){
                        middle_nei_tobeadd.insert(nbs);         
                    }                    
                }
                if(countfornei>20)break;
            } 

            if(tmp_candidate_set.size()==0){                             
                hidden_clus[global_index] = global_index;
                hidden_clu_members[global_index].insert(global_index);
                middle_center_vectors[global_index] = vector;
                for(int x : middle_nei_tobeadd){
                    middle_layer_neighbors[global_index].insert(x);
                    middle_layer_neighbors[x].insert(global_index);
                }
                //small_neibors[global_index] = std::set<int>(); 
            }else{                                                     

                small_neibors[global_index] = std::set<int>(); 
                std::set<int> around = std::set<int>();
                float around_dis;  
                int pre_clu_id = -1;
                for(int tmp_c : tmp_candidate_set){      
              
                    std::uniform_int_distribution<> dis(0, hidden_clu_members[tmp_c].size() - 1);
                    int random_offset = dis(gen);

                    auto it = hidden_clu_members[tmp_c].begin();
                    std::advance(it, random_offset);
                    int start = *it;
                    int min_dis_point = start;
                    float min_dis = compute_l2_distance(vector , hidden_vectors[start]);
                    
                    std::set<int> passed_set =  std::set<int>();
                    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> nearest_neibor; 
                    nearest_neibor.emplace(min_dis,start);
                    while(true){
                        bool flag = true;
                        std::set<int> tobecompute;
                        for(int nei1 : small_neibors[start]){
                            tobecompute.insert(nei1);
                            for(int nei2 : small_neibors[nei1]){
                                tobecompute.insert(nei2);
                            }
                        }
                        for(int s_neibor : tobecompute){    
                            if(passed_set.find(s_neibor)!=passed_set.end()){
                                continue;
                            }
                            passed_set.insert(s_neibor);
                            float res = compute_l2_distance(vector , hidden_vectors[s_neibor]);
                            nearest_neibor.emplace(res,s_neibor);  
                            if(res<min_dis){
                                min_dis = res;
                                min_dis_point = s_neibor;
                                start = s_neibor;
                                flag = false;
                            }
                        }
                        if(flag)break;
                    }
                    float double_dis = 0;
                    if(nearest_neibor.size()>1){
                        double_dis = nearest_neibor.top().first;
                        nearest_neibor.pop();
                        double_dis += nearest_neibor.top().first;
                    }else{
                        double_dis = min_dis*2;
                    }

                    double_dis /= 2;
                    float dis_test;
                    
                    dis_test = alpha/(std::log(1+hidden_clu_members[tmp_c].size())+1);
                    if(double_dis <  dis_test ){  
                        around.insert(tmp_c);   
                        
                        small_neibors[global_index].insert(min_dis_point);
                        small_neibors[min_dis_point].insert(global_index);
                        if(pre_clu_id<0){
                            around_dis = min_dis;
                            pre_clu_id = tmp_c;
                        }
                    }
                }
                if(pre_clu_id==-1){   //notfound
                    hidden_clus[global_index] = global_index;
                    hidden_clu_members[global_index].insert(global_index);
                    middle_center_vectors[global_index] = vector;
                    radius[global_index] =  alpha/(std::log(2)+1) ; //x1;           
              
                }else{
                
               
                    around.erase(pre_clu_id);
                    for(int group_id:around){ 
                            for(int more : upper_layer_neighbors[group_id]){
                                upper_layer_neighbors[more].erase(group_id);
                            }
                            for(int sd : middle_layer_neighbors[group_id]){
                                middle_layer_neighbors[sd].erase(group_id);
                            }
                            ////////////////////////////
                            // std::ofstream ofx3("log.txt",std::ios::app);
                            // ofx3<<"change clu pre:  "<<pre_clu_id<<" from "<<group_id<<std::endl;
                            // ofx3.close();

                            small_neibors[pre_clu_id].insert(group_id);
                            small_neibors[group_id].insert(pre_clu_id);
                            for(int id : hidden_clu_members[group_id]){  
                                hidden_clu_members[pre_clu_id].insert(id);
                                hidden_clus[id] = pre_clu_id;
                                std::uniform_real_distribution<> dis2(0.0, 1.0);
                                double random_value = dis2(gen);
                                if(random_value<0.5){
                                    small_neibors[pre_clu_id].insert(group_id);
                                    small_neibors[group_id].insert(pre_clu_id);
                                }    
                            }
                            for(int nei : middle_layer_neighbors[group_id]){ 
                                if(nei == pre_clu_id){
                                    continue;
                                }
                                middle_layer_neighbors[pre_clu_id].insert(nei);
                                middle_layer_neighbors[nei].insert(pre_clu_id);

                            }
                            
                            float angle_group_id = acos(1.0-radius[group_id]);
                            float angle_dis = acos(1.0-compute_l2_distance(middle_center_vectors[group_id] , middle_center_vectors[pre_clu_id]));     
                            float angle_pre_clu_id= acos(1.0-radius[pre_clu_id]);

                            if( angle_pre_clu_id + angle_dis + angle_group_id > 2*alpha){ 
                                complex_update(group_id,pre_clu_id); 
                            }else{                                           
                            
                                radius[pre_clu_id] = std::max(std::min(angle_group_id,angle_pre_clu_id) + angle_dis, std::max(angle_group_id,angle_pre_clu_id));  

                              
                                for(int x_indice = 0;x_indice<middle_center_vectors[pre_clu_id].size();x_indice++){
                                    middle_center_vectors[pre_clu_id][x_indice] =  (middle_center_vectors[group_id][x_indice] + middle_center_vectors[pre_clu_id][x_indice])/2;
                                }
                                
                            }


                            radius.erase(group_id);
                            hidden_clu_members.erase(group_id);
                            middle_center_vectors.erase(group_id);
                         
                            middle_layer_neighbors.erase(group_id);
                            if(upper_hidden_record.find(group_id) != upper_hidden_record.end()){
                                for(int x : upper_layer_neighbors[group_id]){
                                    upper_layer_neighbors[x].erase(group_id);
                                }
                                upper_layer_neighbors.erase(group_id);
                            }
                            upper_hidden_record.erase(group_id);
                        
                    }
                    
                     if(around.size()==0){
                        float dis1= acos(1.0-around_dis);
                        float dis2= acos(1.0-radius[pre_clu_id]);
                        float dis3=  acos(1.0-(alpha/(std::log(1+hidden_clu_members[pre_clu_id].size())+1)));
                        if(dis2<dis1+dis3){   
                            radius[pre_clu_id] = 1.01*radius[pre_clu_id];       
                        }
                    }

                    hidden_clus[global_index] = pre_clu_id;
                    hidden_clu_members[pre_clu_id].insert(global_index);
                    
                    for(int nb : middle_layer_neighbors[global_index]){
                        middle_layer_neighbors[nb].erase(global_index);
                    }  
                    middle_layer_neighbors.erase(global_index);
                    small_neibors[pre_clu_id].insert(global_index);
                    small_neibors[global_index].insert(pre_clu_id);            
                    if(checkTopic(pre_clu_id)){
                        generateTopic(pre_clu_id , topic_index);
                        return ;
                    }
                    return ;   
                }
            }
          
            if((1.0*(float)upper_hidden_record.size()/hidden_vectors.size())<0.30){
                if(upper_hidden_record.size()>10 && upper_min_dis<topic_lim){   
                    
                }else{
                    std::uniform_real_distribution<float> dis(0.0, 4.0);
                    float random_offset = dis(gen);
                    if(random_offset < upper_min_dis){ 
                        up2upper_layer(hidden_clus[global_index],very_start);
                    }
                }
            }

    }
    void go_to_nearest_topic( TOPIC_index& topic_index,float new_threshold){
        topic_index.set_topic_threshold(new_threshold);
        int num = 0;
        std::set<int> ok_Set;
        std::set<int> passed_set;
        for(auto i : hidden_clu_members){
            if(i.second.size()>1){
                if(passed_set.find(i.first)!=passed_set.end())continue;
                passed_set.insert(i.first);
                int chosen = i.first;
                topic_index.begin_add_topic();   
                for(int x : hidden_clu_members[chosen]){ //
                    topic_index.GTopic(hidden_layer_line[x] , hidden_vectors[x]);
                    hidden_vectors.erase(x);
                    ok_Set.insert(x);
                }       
                topic_index.GTopic(hidden_layer_line[chosen],hidden_vectors[chosen]);
                topic_index.end_add_topic(radius[chosen] , false);    
            }
        }
        if(test_f)std::cout<<"num: "<<num<<std::endl;
        for(auto i : hidden_vectors){
            num++;
            if(test_f)std::cout<<"i.second size: "<<i.second.size()<<std::endl;
            if(i.second.size()!=1024)continue;
            int res = topic_index.find_topic_HNSW(i.second,hidden_layer_line[i.first]);
            if(res>-1){
                ok_Set.insert(i.first);
            }
        }
        for(int i : ok_Set){
            hidden_vectors.erase(i);
            small_neibors.erase(i);
            hidden_layer_line.erase(i);
        }
        std::cout<<"num: "<<num<<std::endl;
    }

    void become_topic(TOPIC_index& topic_index){
        for(auto i : hidden_vectors){
            
            topic_index.find_topic_HNSW(i.second,hidden_layer_line[i.first]);
        }
    }

    void remake(){
        
        std::set<int> okset;
        int random_value = global_index-50000;
        for(auto i : hidden_vectors){   
            if(i.first < random_value){
                okset.insert(i.first);
            }   
        }
        for(int i:okset){
            //justEraseOne(i);
            eraseOne(hidden_clus[i]);
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
#endif // ONLINE_CLUSTER_H



