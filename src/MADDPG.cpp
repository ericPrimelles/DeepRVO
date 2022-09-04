#include "MADDPG.h"
#include <iostream>
#include <fstream>

using std::cout, std::endl;

MADDPG::MADDPG(Environment *sim, int64_t Ain_dims, int64_t Aout_dims, std::vector<int64_t> Ah_dims, int64_t Cin_dims,
               int64_t Cout_dims, std::vector<int64_t> Ch_dims, size_t scenario, float alpha,
               float beta, size_t fc1, size_t fc2, size_t T, float gamma, float tau, float ou_sigma,
               std::string path)
{
    this->env = sim;
    this->Ain_dims = Ain_dims;
    this->Aout_dims = Aout_dims;
    this->Cin_dims = Cin_dims;
    this->Cout_dims = Cout_dims;
    this->n_agents = this->env->getNAgents();
    this->scenario = scenario;
    this->alpha = alpha;
    this->beta = beta;
    this->fc1 = fc1;
    this->fc2 = fc2;
    this->ou_sigma = ou_sigma;
    this->gamma = gamma;
    this->tau = tau;
    this->path = path;

    if (torch::cuda::is_available())
        this->device = torch::Device(torch::kCUDA);

    // this->agents.reserve(this->n_agents);

    for (size_t i = 0; i < n_agents; i++)
    {
        this->agents.push_back(new DDPGAgent(this->Ain_dims, this->Aout_dims, Ah_dims, this->Cin_dims, this->Cout_dims, Ch_dims,
                                             this->gamma, this->tau));
        this->agents[i]->updateParameters(1.0f);
    }
}

MADDPG::~MADDPG()
{
}

void MADDPG::saveCheckpoint()
{
    cout << "...saving chekpoint..." << endl;

    for (size_t i = 0; i < this->n_agents; i++)
    {
        agents[i]->saveModel(this->path, i);
    }
}

void MADDPG::loadCheckpoint()
{
    cout << "...loading chekpoint..." << endl;

    for (size_t i = 0; i < this->n_agents; i++)
    {
        agents[i]->loadModel(this->path, i);
    }
}

torch::Tensor MADDPG::chooseAction(torch::Tensor obs, bool use_rnd, bool use_net)
{
    cout << "From MADDPG" << this->n_agents << endl;
    torch::Tensor actions = torch::zeros({(int64_t)this->n_agents, (int64_t)this->env->getActionSpace()}, torch::dtype(torch::kFloat32)).to(device);

    if (use_net)
    {
        for (size_t i = 0; i < this->n_agents; i++)
        {
            actions = this->eval(obs).to(device);

            // std::cout << i << std::endl;
        }
    }
    if (use_rnd)
    {
        actions += this->ou_sigma * torch::rand(actions.sizes()).to(device);
    }
    // std::cout << actions[0].sizes() << endl;
    return actions;
}

void MADDPG::Train(size_t k_epochs, size_t T)
{

    ReplayBuffer::Buffer *memory = new ReplayBuffer::Buffer();
    ReplayBuffer::Transition a;
    vector<ReplayBuffer::Transition> sampledTrans;

    for (size_t i = 0; i < k_epochs; i++)
    {
        env->reset();
        float avg_reward = 00.f;
        float step_rewards = 00.f;
        for (size_t j = 0; j < T || env->isDone(); j++)
        {
            cout << "Epoch: " << i << "/" << k_epochs << " ";
            std::cout << "TimeStep:" << j << "/" << T << " Rewards:"
                      << "    " << avg_reward << std::endl;
            a.obs = env->getObservation().to(device);
            a.actions = this->chooseAction(a.obs, true, memory->ready()).to(device);
            a.rewards = env->step(a.actions).to(device);
            a.obs_1 = env->getObservation().to(device);
            a.done = env->isDone();
            memory->storeTransition(a);
            env->render(j, i);

            cout << "Global time:" << env->getGlobalTime() << endl;

            sampledTrans = memory->sampleBuffer();
            if (memory->ready())
                this->learn(sampledTrans);

            step_rewards += torch::mean(a.rewards).item<float>();
            avg_reward = step_rewards / (j + 1);
            // sampledTrans = memory->sampleBuffer();
        }
        if (i % 10 == 0)
        {
            std::cout << "Saving..." << std::endl;
            this->saveCheckpoint();
        }

        std::ofstream write;
        write.open("rewards.txt", std::ios::out | std::ios::app);
        if (write.is_open())
        {
            write << avg_reward << "\n";
        }

        write.close();
    }
}

void MADDPG::Test(size_t epochs)
{

    /*std::cout << "Testing" << std::endl;
    this->loadCheckpoint();
    for (size_t i = 0; i < epochs; i++)
    {

        this->env->reset();
        for (size_t j = 0; j < this->T; j++)
        {
            this->env->step(this->chooseAction(this->env->getObservation(), false, true));
            // this->chooseAction(this->env->getObservation());
        }
    }*/
}

void MADDPG::visualize()
{
    for (size_t i = 0; i < this->n_agents; i++)
    {
        std::cout << this->env->getAgentPos(i) << "--";
    }

    std::cout << std::endl;
}

void MADDPG::learn(vector<ReplayBuffer::Transition> sampledTrans)
{
    // Cut from here
    for (size_t agent = 0; agent < this->n_agents; agent++)
    {

        this->agents[agent]->a_optim.zero_grad();
        this->agents[agent]->c_optim.zero_grad();
        float memsize_scale = 1.0f / static_cast<float>(this->batch_size);
        torch::Tensor q_loss = torch::zeros({1}).to(device);
        torch::Tensor a_loss = torch::zeros({1}).to(device);

        for (auto &t : sampledTrans)
        {
            torch::Tensor target;
            torch::Tensor ret;

            if (!t.done)
            {

                ret = (this->gamma * agents[agent]->target_c_n(torch::cat({t.obs_1.flatten(), this->eval(t.obs_1).flatten().to(device)}).to(device)).detach()).to(device);
                target = (t.rewards[agent] + ret).to(device);
            }
            else
            {

                target = t.rewards[agent] * torch::ones({2}).to(device);
            }

            torch::Tensor seg_loss = this->agents[agent]->target_c_n(torch::cat({t.obs.flatten(), t.actions.flatten()}).detach()).to(device) - target;

            q_loss += memsize_scale * seg_loss * seg_loss;
        }
        q_loss.backward();
        this->agents[agent]->c_optim.step();

        //*traj_q_loss += q_loss.item<float>();
        for (auto &t : sampledTrans)
        {
            a_loss -= memsize_scale * this->agents[agent]->c_n(torch::cat({t.obs.flatten(), this->eval(t.obs).flatten().to(device)}).detach()).to(device);
        }
        a_loss.backward();
        this->agents[agent]->a_optim.step();
        //*traj_a_loss += a_loss.item<float>();
        this->agents[agent]->updateParameters(tau);
    }
}

torch::Tensor MADDPG::eval(torch::Tensor obs)
{

    torch::Tensor evaluation = torch::empty({(int64_t)this->getNAgents(), (int64_t )this->env->getActionSpace()});

    for (size_t agentx = 0; agentx < this->getNAgents(); agentx++)
    {
        evaluation[agentx] = this->agents[agentx]->target_a_n(obs[agentx]);
    }

    return evaluation;
}