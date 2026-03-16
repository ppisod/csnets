using csnets.Neural;
namespace csnets.Neural.Initializations;

/// <summary>
/// Naive uniform initialization. Weights in [-1, 1].
/// </summary>
public class RandomInit : IInitialization {
    public Weight[] InitWeights ( Random random, int inputSize ) {
        Weight[] weights = new Weight[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = new Weight
            {
                gradients = [],
                value = (float) ( random.NextDouble () * 2 - 1 )
            };
        }
        return weights;
    }
}
