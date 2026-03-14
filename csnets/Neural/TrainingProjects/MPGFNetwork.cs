using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public class MPGFNetwork (
    int gen_C, int gen_M,
    int datasetSize,
    int layers,
    int minNeurons,
    int maxNeurons,
    float momentum
) : INet {
    public FeedForwardNet net { get; set; } = new ( 2, layers, 1, momentum, minNeurons, maxNeurons );
    public MidPointGenFunc gen { get; set; } = new () { c = gen_C, m = gen_M };

    public int datasetSize = datasetSize;


    public void Train ( int epochs ) {
        List<MidPointGenFuncOutput> dataset = gen.gen(datasetSize);
        for (int i = 0; i < epochs; i++)
        {
            foreach (var datapoint in dataset)
            {
                net.Train <ReLU, MeanSquaredError> ([datapoint.in_A, datapoint.in_B], [datapoint.out_M], 0.001f);
            }
        }
    }

    public float[] Run ( float[] inputs ) {
        return net.Run <ReLU> (inputs);
    }

    public float Run ( ) {
        List <MidPointGenFuncOutput> dataset = gen.gen ( datasetSize );
        List <float> rs = [];
        foreach (var datapoint in dataset)
        {

            rs.Add((datapoint.out_M - Run([datapoint.in_A, datapoint.in_B])[0]) / datapoint.out_M);
        }
        return rs.Average ();
    }
}