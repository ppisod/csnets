using csnets.Neural.Activations;
namespace csnets.Neural;

public class Neuron : INeuron {

    public float[] weights { get; set; }
    public float bias { get; set; }
    public float[] weightAccumulator { get; set; }
    public float biasAccumulator { get; set; }
    public bool batching { get; set; }


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
        if (inputs.Length != weights.Length) throw new Exception ("Input size does not match weight size.");
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
    /// <param name="updateWeights">if false, don't update weights AND return weight gradient</param>'
    /// <returns>blame for each input (to pass to the previous layer)</returns>
    public virtual BackPropResult BackProp <A> ( float[] inputs, float blame, float learningRate, bool isLastLayer = false, bool updateWeights = true) where A : IActivation {
        BackPropResult result;
        float myFinalBlame;
        if (isLastLayer) {
            myFinalBlame = blame;
            result.blame = blame;
        } else {
            // if neuron didn't contribute to output because of the Activation func
            // then it should not be blamed for the output
            // that's why we see if Activation function fucked up the output
            var rawOutput = ForwardPass <A> ( inputs, false );
            myFinalBlame = A.Df ( rawOutput ) * blame;
            result.blame = myFinalBlame;
        }

        // Change activations in proportion to weights. (We cannot change it, therefore we return values as blame)
        float[] inputBlame = new float[inputs.Length];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputBlame[i] = myFinalBlame * weights[i];
        }
        result.nextLayerBlame = inputBlame;

        float[] weightGradients = new float[inputs.Length];

        // Change weights in proportion to activations.
        for (var i = 0; i < inputs.Length; i++)
        {
            if (!updateWeights)
            {
                weightGradients[i] = - ( myFinalBlame * inputs[i] * learningRate );
                continue;
            }
            weights[i] -= myFinalBlame * inputs[i] * learningRate;
        }

        result.weightGradient = weightGradients;

        // Update bias
        bias -= myFinalBlame * learningRate;

        result.biasGradient = -myFinalBlame * learningRate;

        return result;
    }

    public virtual void AccumulateGradients ( float[] inputs, float realBlame ) {
        for (int i = 0; i < inputs.Length; i++)
        {
            weightAccumulator[i] += realBlame * inputs[i];
        }

        biasAccumulator += realBlame;
    }
    public virtual void ApplyGradients ( float learningRate, int batches ) {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] -= (weightAccumulator[i] / batches) * learningRate;
            weightAccumulator[i] = 0;
        }
        bias -= (biasAccumulator / batches) * learningRate;
        biasAccumulator = 0;
    }
}