
using csnets.Neural.TrainingProjects;

MPGFNetwork network = new (1000, 3, 3, 5, 10, 0.9f);
network.Train (10);
