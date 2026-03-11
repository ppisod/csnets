using csnets.Neural.Activations;
namespace csnets.Neural;

public class Neuron : INeuron {

    public float[] weights { get; set; }
    public float bias { get; set; }


    /// <summary>
    /// Naively initializes a neuron with a given Random.
    /// Weight: -1 to 1f
    /// Bias: 0
    /// </summary>
    public Neuron ( Random random, int inputSize ) {
        weights = new float[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = ( (float) random.NextDouble () * 2 - 1 );
        }
        bias = 0;
    }

    public virtual float ForwardPass <A> ( float[] inputs, bool activate ) where A : IActivation {
        if (inputs.Count != weights.Count) throw new Exception ("Input size does not match weight size.");
        float sum = 0;
        for (var index = 0; index < inputs.Length; index++)
        {
            var input = inputs[index];
            var weight = weights[index];
            sum += input * weight;
        }
        sum += bias;

        if (activate) return A.F (sum);
        return sum;
    }

    /// <summary>
    /// Backpropagates blame through this neuron.
    /// Scales blame by the activation derivative, updates weights and bias,
    /// and returns the blame to pass to the previous layer.
    /// ~ opus
    /// </summary>
    /// <param name="inputs">the inputs this neuron received during the forward pass</param>
    /// <param name="blame">the blame assigned to this neuron from the next layer</param>
    /// <param name="learningRate">how aggressively to update weights</param>
    /// <param name="isLastLayer">if true, skip activation derivative (output layer has no activation)</param>
    /// <returns>blame for each input (to pass to the previous layer)</returns>
    public virtual float[] BackProp <A> ( float[] inputs, float blame, float learningRate, bool isLastLayer = false ) where A : IActivation {
        float myFinalBlame;
        if (isLastLayer) {
            myFinalBlame = blame;
        } else {
            // if neuron didn't contribute to output because of the Activation func
            // then it should not be blamed for the output
            // that's why we see if Activation function fucked up the output
            var rawOutput = ForwardPass <A> ( inputs, false );
            myFinalBlame = A.Df ( rawOutput ) * blame;
        }

        // Change activations in proportion to weights. (We cannot change it, therefore we return values as blame)
        float[] inputBlame = new float[inputs.Length];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputBlame[i] = myFinalBlame * weights[i];
        }

        // Change weights in proportion to activations.
        for (var i = 0; i < inputs.Length; i++)
        {
            weights[i] -= myFinalBlame * inputs[i] * learningRate;
        }

        // Update bias
        bias -= myFinalBlame * learningRate;

        return inputBlame;
    }
}