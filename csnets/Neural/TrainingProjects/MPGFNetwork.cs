using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public class MPGFNetwork (
    int datasetSize,
    int epochs,
    int layers,
    int minNeurons,
    int maxNeurons,
    float momentum
) : INet {
    public FeedForwardNet net { get; set; } = new ( 2, layers, 1, momentum, minNeurons, maxNeurons );

    public int datasetSize = datasetSize;


    public void Train ( int epochs ) {
        MidPointGenFunc gen = new ()
        {
            c = 7,
            m = 1,
        };
        List<MidPointGenFuncOutput> dataset = gen.gen(datasetSize);
        for (int i = 0; i < epochs; i++)
        {
            foreach (var datapoint in dataset)
            {
                net.Train <ReLU, MeanSquaredError> ([datapoint.in_A, datapoint.in_B ], [datapoint.out_M], 0.001f, true);
            }
        }
    }

    public float[] Run ( float[] inputs ) {
        return net.Run <ReLU> (inputs);
    }
}