
using csnets.Neural.TrainingProjects;

MNISTNetwork network = new MNISTNetwork ( [16, 16], 0.9f, 0.001f);
network.Train (5);

Console.Out.WriteLine($"accuracy: {network.Run()}"); // YAY