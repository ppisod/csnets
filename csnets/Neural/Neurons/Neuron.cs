using csnets.Neural.Activations;
namespace csnets.Neural;

public class Neuron {
    public List<float> weights;
    public float bias;

    /// <summary>
    /// Naively initializes a neuron with a given Random.
    /// Weight: -1 to 1f
    /// Bias: 0
    /// </summary>
    public Neuron ( Random random, int inputSize ) {
        weights = [];
        for (int i = 0; i < inputSize; i++)
        {
            weights.Add ( (float) random.NextDouble () * 2 - 1 );
        }
        bias = 0;
    }

    public float ForwardPass <A> ( List <float> inputs, bool activate ) where A : IActivation {
        if (inputs.Count != weights.Count) throw new Exception ("Input size does not match weight size.");
        float sum = 0;
        for (var index = 0; index < inputs.Count; index++)
        {
            var input = inputs[index];
            var weight = weights[index];
            sum += input * weight;
        }
        sum += bias;

        if (activate) return A.F (sum);
        return sum;
    }
}