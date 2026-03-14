using csnets.Neural.Activations;

namespace csnets.Neural;

public class MomentalNeuron (
    Random random,
    int inputSize,
    float momentum
) : Neuron ( random, inputSize ) {

    public readonly float momentum = momentum;
    public float[] weightVelocities = new float[inputSize];
    public float biasVelocity = 0;

    public float[] weightGradAccumulator = new float[inputSize];
    public float biasGradAccumulator = 0;

    public bool batching = false;

    public override void ApplyGradients ( float learningRate, int batches ) {
        for (int i = 0; i < weights.Length; i++)
        {
            float avgGrad = weightGradAccumulator[i] / batches;
            weightVelocities[i] = ( momentum * weightVelocities[i] ) - ( avgGrad * learningRate );
            weights[i] += weightVelocities[i];
            weightGradAccumulator[i] = 0;
        }

        float avgBiasGrad = biasGradAccumulator / batches;
        biasVelocity = ( momentum * biasVelocity ) - ( avgBiasGrad * learningRate );
        bias += biasVelocity;
        biasGradAccumulator = 0;
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
            var inp = inputs[index];
            pastBlames[index] = ( realBlame * weights[index] );
        }

        result.nextLayerBlame = pastBlames;

        float[] weightGradients = new float[inputs.Length];
        // change weight proportional to activation
        for (var index = 0; index < inputs.Length; index++)
        {
            var inp = inputs[index];
            weightVelocities[index] = ( momentum * weightVelocities[index] ) - ( inp * realBlame * learningRate );
            weights[index] += weightVelocities[index];
            weightGradients[index] = weightVelocities[index];
        }

        result.weightGradient = weightGradients;

        // change bias
        biasVelocity = ( momentum * biasVelocity ) - ( realBlame * learningRate );
        bias += biasVelocity;
        result.biasGradient = -biasVelocity;

        return result;
    }

}