
using csnets.Neural.Optimizers;
using csnets.Neural.TrainingProjects;

MNISTNetwork network = new MNISTNetwork ( [128], new AdamOptimizer ( ), 0.0001f);
network.Train (5);

Console.Out.WriteLine($"accuracy: {network.Run()}"); // YAY
