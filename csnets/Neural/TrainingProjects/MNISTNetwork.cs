using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;
using csnets.Neural.Optimizers;

namespace csnets.Neural.TrainingProjects;

public class MNISTNetwork : INet {

    public float learnRate;
    public int batchSize;

    public MNISTNetwork ( int[] layers, IOptimizer optimizer, float learnRate, int batchSize = 2048 ) {
        mnistData = new MNIST ();
        mnistData.Load ();
        net = new FeedForwardNet (mnistData.PixelCount, optimizer, layers, mnistData.LabelCount);
        this.learnRate = learnRate;
        this.batchSize = batchSize;
    }

    public MNIST mnistData { get; set; }

    public FeedForwardNet net { get; set; }
    private Random rng = new ();
    public void Train ( int epochs ) {
        int totalSamples = mnistData.TrainSet.Count;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int processed = 0;
            float lastLoss = 0;
            Console.WriteLine ( $"Epoch {epoch + 1}/{epochs}" );
            Shuffle ( mnistData.TrainSet );
            foreach (var batch in mnistData.TrainSet.Chunk ( batchSize ))
            {
                float[][] inp = batch.Select ( image => image.Pixels ).ToArray ();
                float[][] targ = batch.Select ( image => image.LabelOneHot ).ToArray ();
                net.Train <ReLU, SoftmaxCrossEntropy> ( inp, targ, learnRate );
                processed += inp.Length;
                var outputs = net.Run <ReLU> ( inp[^1] );
                lastLoss = SoftmaxCrossEntropy.Calculate ( outputs, targ[^1] );
                net.plot.AddLoss ( lastLoss );
                FeedForwardNet.PrintProgress ( processed, totalSamples, lastLoss );
            }
            net.plot.Save ();
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
            int predicted = 0;
            int expected = 0;
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > output[predicted]) predicted = i;
                if (image.LabelOneHot[i] > image.LabelOneHot[expected]) expected = i;
            }
            if (predicted == expected) correct++;
        }
        return (float) correct / mnistData.TestSet.Count;
    }

    private void Shuffle <T> ( List<T> list ) {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next ( i + 1 );
            ( list[i], list[j] ) = ( list[j], list[i] );
        }
    }

}
