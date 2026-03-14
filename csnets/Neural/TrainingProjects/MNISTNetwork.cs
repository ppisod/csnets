using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public class MNISTNetwork : INet {

    public float learnRate;

    public MNISTNetwork ( int[] layers, float momentum, float learnRate ) {
        mnistData = new MNIST ();
        mnistData.Load ();
        net = new FeedForwardNet (mnistData.PixelCount, momentum, layers, mnistData.LabelCount);
        this.learnRate = learnRate;
    }

    public MNIST mnistData { get; set; }

    public FeedForwardNet net { get; set; }
    public void Train ( int epochs ) {
        for (int i = 0; i < epochs; i++)
        {
            mnistData.TrainSet.ForEach ( image =>
                {
                    net.Train <ReLU, MeanSquaredError> (
                        image.Pixels,
                        image.LabelOneHot,
                        learnRate,
                        true
                    );
                }
            );
        }
    }
    public float[] Run ( float[] inputs ) {
        return net.Run <ReLU> ( inputs );
    }
    public float Run ( ) {
        int correct = 0;
        foreach (var image in mnistData.TestSet)
        {
            float[] output = Run ( image.Pixels );
            bool match = true;
            for (int i = 0; i < output.Length; i++)
            {
                if (MathF.Abs ( output[i] - image.LabelOneHot[i] ) > 0.05f)
                {
                    match = false;
                    break;
                }
            }
            if (match) correct++;
        }
        return (float) correct / mnistData.TestSet.Count;
    }

}