
using csnets.Neural.TrainingProjects;

MPGFNetwork network = new (7, 1, 1000, 3, 3, 5, 0.9f);
network.Train (10);

Console.Out.WriteLine($"ERROR: {network.Run()}"); // YAY