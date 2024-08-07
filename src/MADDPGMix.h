#ifndef MADDPGMIX_H
#define MADDPGMIX_H

#pragma once


#include "torch/torch.h"
#include "Environment.h"
#include "DDPGAgent.h"
#include "Buffer.h"
#include <vector>
#include <string>


class MADDPGMix
{
public:
    MADDPGMix(Environment *sim,int64_t Ain_dims, int64_t Aout_dims, std::vector<int64_t> Ah_dims, int64_t Cin_dims,
           int64_t Cout_dims, std::vector<int64_t> Ch_dims, size_t scenario, float alpha=0.01, 
           float beta=0.01, size_t fc1=64, size_t fc2=64, size_t T=10, float gamma=0.99, float tau=0.01, float ou_sigma=0.4f, 
           std::string path="");
    void saveCheckpoint();
    void loadCheckpoint();
    torch::Tensor chooseAction(torch::Tensor obs, bool use_rnd = true, bool use_net = true);
    void Train(size_t, size_t);
    void Test(size_t epochs);
    size_t getNAgents(){return n_agents;}
    inline DDPGAgent* getAgent(size_t i){ return agents[i];}

    ~MADDPGMix();

private:

    //Private Methods
    void visualize();
    void learn(vector<ReplayBuffer::Transition>);
    Vector2 doOrca(size_t agent);
    torch::Tensor eval(torch::Tensor obs);
    // Parameters
    int64_t Ain_dims, Aout_dims, Cin_dims, Cout_dims;
    size_t scenario, batch_size, n_agents;
    float alpha, beta, fc1, fc2, gamma, tau, ou_sigma;
    std::string path;
    Environment *env;
    std::vector<DDPGAgent*> agents;
    torch::Device device = torch::Device(torch::kCPU);
    size_t agent_1, agent_2;

};

#endif