using csnets.Neural.Activations;
using csnets.Neural.Initializations;

namespace csnets.Neural;

public class MomentalNeuron (
    Random random,
    int inputSize,
    float momentum,
    IInitialization init
) : Neuron ( random, inputSize, init ) {

    public readonly float momentum = momentum;
    public float[] weightVelocities = new float[inputSize];
    public float biasVelocity = 0;

    public override void ApplyGradients ( float learningRate ) {
        for (int i = 0; i < weights.Length; i++)
        {
            float avgGrad = weights[i].AverageGradient ();
            weightVelocities[i] = ( momentum * weightVelocities[i] ) - ( avgGrad * learningRate );
            weights[i].value += weightVelocities[i];
            weights[i].gradients.Clear ();
        }

        float avgBiasGrad = bias.AverageGradient ();
        biasVelocity = ( momentum * biasVelocity ) - ( avgBiasGrad * learningRate );
        bias.value += biasVelocity;
        bias.gradients.Clear ();
    }

    public override BackPropResult BackProp <A> (
        float[] inputs,
        float blame,
        float learningRate,
        bool isLastLayer = false,
        bool updateWeights = true
    ) {
        BackPropResult result;
        float realBlame;
        if (isLastLayer)
        {
            realBlame = blame;
        }
        else
        {
            float unactivatedOut = ForwardPass <A> ( inputs, false );
            realBlame = blame * A.Df ( unactivatedOut );
        }

        result.blame = realBlame;

        float[] pastBlames = new float[inputs.Length];
        for (var index = 0; index < inputs.Length; index++)
        {
            pastBlames[index] = realBlame * weights[index].value;
        }

        result.nextLayerBlame = pastBlames;

        float[] weightGradients = new float[inputs.Length];

        if (!updateWeights)
        {
            // Batch mode: accumulate gradients
            for (var index = 0; index < inputs.Length; index++)
            {
                weights[index].AddGradient ( realBlame * inputs[index] );
            }
            bias.AddGradient ( realBlame );
            result.weightGradient = weightGradients;
            result.biasGradient = 0;
        }
        else
        {
            // SGD mode: update weights immediately via momentum
            for (var index = 0; index < inputs.Length; index++)
            {
                var inp = inputs[index];
                weightVelocities[index] = ( momentum * weightVelocities[index] ) - ( inp * realBlame * learningRate );
                weights[index].value += weightVelocities[index];
                weightGradients[index] = weightVelocities[index];
            }

            result.weightGradient = weightGradients;

            biasVelocity = ( momentum * biasVelocity ) - ( realBlame * learningRate );
            bias.value += biasVelocity;
            result.biasGradient = -biasVelocity;
        }

        return result;
    }

}
