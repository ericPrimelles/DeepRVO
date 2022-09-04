#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "RVO.h"
#include "Environment.h"
#include "MADDPG.h"
#include "MADDPGMix.h"
#include "Buffer.h"

using namespace std;
using namespace RVO;

// Environment global varaiables
size_t Agents = 10;
float timestep = 0.25f;
float neighbor_dist = 1.0f;
size_t max_neig = Agents;
float time_horizont = 10.0f;
float time_horizont_obst = 20.0f;
float radius = 2.0f;
float max_speed = 3.5f;

// Train Hyperparameters
size_t k_epochs = 1000;
size_t T = 10000;
size_t batch_size = 256;

size_t scenario = 1, sidesize = 25;
float traj_reward = 0.0f;
float traj_q_loss = 0.0f;
float traj_a_loss = 0.0f;
float avg_reward = 0.0f;
float step_rewards = 0.0f;
Environment *env = new Environment(Agents, timestep, neighbor_dist, max_neig, time_horizont, time_horizont_obst, radius, max_speed);
torch::Device device(torch::kCPU);

// Execution parameters
bool train = true;
// Exec functions

void Train(Environment *env, MADDPG program);
void Test (Environment * env, MADDPG program, size_t n_epochs);

// Visualization func
void InitGL(void);
void reshape(int width, int height); // OpenGL function
void idle(void);
void renderBitmapString(float x, float y, void *font, char *string);
void updateVisualization();

int main(int argc, char **argv)
{
   env->make(1, false);
   MADDPGMix program(env, 4, 2, {32, 16, 8}, 4*env->getNAgents() + 2*env->getNAgents(), 1, {32, 16, 8}, 0);

   /*glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(800, 600);
   glutCreateWindow("Deep Nav");
   glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
   InitGL();
   glutDisplayFunc(updateVisualization);
   glutReshapeFunc(reshape);
   glutIdleFunc(idle);*/
   
   // DEVICE setting
   
   if (torch::cuda::is_available()){
      cout << "CUDA is available" << endl;
      device = torch::Device(torch::kCUDA);
      /*for(int i = 0; i < program.getNAgents(); i ++){
      program.getAgent(i)->a_n->to(device);
      program.getAgent(i)->c_n->to(device);
      program.getAgent(i)->target_a_n->to(device);
      program.getAgent(i)->target_c_n->to(device);*/
   //}
   } else{
      cout << "CUDA isn't available. Using CPU" << endl;
   }
   
   
   program.Train(k_epochs, T);

  
   
   return 0;
}


void Test (Environment * env, MADDPG program, size_t n_epochs){
   
   /*program.loadCheckpoint();
   
   for(int i = 0; i < n_epochs; i++){
      env->reset();
      for (int step = 0; step < T || env->isDone(); step ++){
         auto obs = env->getObservation();
         //auto act = program.chooseAction(obs, false, true);
         auto rwds = env->step(act);
         env->render(i, 0);
      }
   }*/
}
/*void InitGL(void) // OpenGL function
{

   glShadeModel(GL_SMOOTH);              // Enable Smooth Shading
   glClearColor(0.5f, 0.5f, 0.5f, 0.5f); // Black Background
   glClearDepth(1.0f);                   // Depth Buffer Setup
   glEnable(GL_DEPTH_TEST);              // Enables Depth Testing
   glDepthFunc(GL_LEQUAL);               // The Type Of Depth Testing To Do
   glEnable(GL_COLOR_MATERIAL);
   glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void reshape(int width, int height) // OpenGL function
{
   glViewport(0, 0, width, height);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   float q = width / (float)height;

   switch (scenario)
   {

   case 0:
      break;
   case 1:
      glOrtho(-10, sidesize + 10, -10, sidesize + 10, -2, 20);
      break;

   case 2:
      gluOrtho2D(-q * 45, q * 45, -45, 45);
      break;

   case 3:
      gluOrtho2D(-q * 20, q * 20, 10, 50);
      break;

   case 4:
      gluOrtho2D(-q * 25, q * 25, -25, 25);
      break;

   case 5:
      glOrtho(-20, 10, -10, 10, -2, 20);
      break;

   case 6:
      gluOrtho2D(-q * 15, q * 15, -15, 15);
      break;

   case 7:
      glOrtho(-30, 30, -20, 20, -2, 20);
      break;
   case 8:
      glOrtho(-10, sidesize + 10, -10, sidesize + 10, -2, 20);
      break;

   case 9:
      glOrtho(-20, 10, -10, 10, -2, 20);
      break;

   case 10:
      glOrtho(-5, 35, -10, 20, -5, 35); // gluOrtho2D(-q*45, q*45, -45, 45 );
      break;
   }

   glMatrixMode(GL_MODELVIEW);
}

void idle(void) // OpenGL function
{
   glutPostRedisplay();
}

void renderBitmapString(float x, float y, void *font, char *string)
{
   char *c;
   glRasterPos2f(x, y);
   for (c = string; *c != '\0'; c++)
   {
      glutBitmapCharacter(font, *c);
   }
}

void updateVisualization()
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   for (size_t i = 0; i < env->getNAgents(); i++)
   {
      glPushMatrix();
      glTranslatef(env->getAgentPos(i).x(), env->getAgentPos(i).y(), 0.0f);

      int num = i;
      char buffer[10] = {'\0'};
      sprintf(buffer, "%d", num); //%d is for integers

      renderBitmapString(0.3, 0.3, GLUT_BITMAP_TIMES_ROMAN_24, buffer);

      glColor3f(0.4f, 0.9f, 0.0f); // greenish color for the agents

      glutSolidSphere(0.5f, 8, 8);
      glDisable(GL_LIGHTING);
      glColor3f(0, 0, 0);

      glBegin(GL_LINES);
      glVertex3f(0.f, 0.f, 1.0f);
      RVO::Vector2 pfrev = env->getAgentPrefVel(i);
      glVertex3f(pfrev.x(), pfrev.y(), 1.0f);
      glEnd();

      glColor3f(0, 0, 0);

      /*if ((indexMostSimilar[i] >= 0) && (i == agentViewed))
      {
          glBegin(GL_LINES);
          glVertex3f(0.f, 0.f, 1.0f);
          RVO::Vector2 simi = RVO::Vector2(sim->getAgentPosition(bestSimilarNeigh[i]) - sim->getAgentPosition(i));
          glVertex3f(simi.x(), simi.y(), 1.0f);
          glEnd();
      }*/
     /* glPopMatrix();
   }
}*/
