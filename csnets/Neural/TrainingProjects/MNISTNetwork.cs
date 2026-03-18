using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;
using csnets.Neural.Optimizers;

namespace csnets.Neural.TrainingProjects;

public class MNISTNetwork : INet {

    public float learnRate;
    public int batchSize = 2048;

    public MNISTNetwork ( int[] layers, IOptimizer optimizer, float learnRate ) {
        mnistData = new MNIST ();
        mnistData.Load ();
        net = new FeedForwardNet (mnistData.PixelCount, optimizer, layers, mnistData.LabelCount);
        this.learnRate = learnRate;
    }

    public MNIST mnistData { get; set; }

    public FeedForwardNet net { get; set; }
    public void Train ( int epochs ) {
        for (int i = 0; i < epochs; i++)
        {
            foreach (var batch in mnistData.TrainSet.Chunk ( batchSize ))
            {
                float[][] inp = batch.Select ( image => image.Pixels ).ToArray ();
                float[][] targ = batch.Select ( image => image.LabelOneHot ).ToArray ();
                net.Train <ReLU, CrossEntropyLoss> ( inp, targ, learnRate, true );
            }
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
