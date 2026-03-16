using csnets.Neural.Activations;
using csnets.Neural.Initializations;
namespace csnets.Neural;

public class Neuron : INeuron {

    public Weight[] weights { get; set; }
    public Weight bias { get; set; }
    public bool batching { get; set; }


    public Neuron ( Random random, int inputSize, IInitialization init ) {
        weights = init.InitWeights ( random, inputSize );
        bias = new Weight { value = 0 };
    }

    public virtual float ForwardPass <A> ( float[] inputs, bool activate ) where A : IActivation {
        if (inputs.Length != weights.Length) throw new Exception ("Input size does not match weight size.");
        float sum = 0;
        for (var index = 0; index < inputs.Length; index++)
        {
            sum += inputs[index] * weights[index].value;
        }
        sum += bias.value;

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
            inputBlame[i] = myFinalBlame * weights[i].value;
        }
        result.nextLayerBlame = inputBlame;

        float[] weightGradients = new float[inputs.Length];

        // Change weights in proportion to activations.
        for (var i = 0; i < inputs.Length; i++)
        {
            if (!updateWeights)
            {
                weights[i].AddGradient ( myFinalBlame * inputs[i] );
                continue;
            }
            weights[i].value -= myFinalBlame * inputs[i] * learningRate;
        }

        result.weightGradient = weightGradients;

        if (!updateWeights)
        {
            bias.AddGradient ( myFinalBlame );
        }
        else
        {
            bias.value -= myFinalBlame * learningRate;
        }

        result.biasGradient = -myFinalBlame * learningRate;

        return result;
    }

    public virtual void ApplyGradients ( float learningRate ) {
        foreach (var weight in weights)
        {
            weight.ApplyGradients ( learningRate );
        }
        bias.ApplyGradients ( learningRate );
    }
}
