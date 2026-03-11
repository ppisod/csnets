namespace csnets.Neural;

public class MomentalNeuron (
        Random random,
        int inputSize,
        float momentum
    ) : Neuron ( random, inputSize ) {

    public readonly float momentum = momentum;
    public float[] weightVelocities = new float[inputSize];
    public float biasVelocity = 0;

    public override float[] BackProp <A> (
        float[] inputs,
        float blame,
        float learningRate,
        bool isLastLayer = false
    ) {
        // scale blame
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

        // change activation proportional to weight
        float[] pastBlames = new float[inputs.Length];
        for (var index = 0; index < inputs.Length; index++)
        {
            var inp = inputs[index];
            pastBlames[index] = (realBlame * weights[index]);
        }

        // change weight proportional to activation
        for (var index = 0; index < inputs.Length; index++)
        {
            var inp = inputs[index];
            weightVelocities[index] = ( momentum * weightVelocities[index] ) - ( inp * realBlame * learningRate );
            weights[index] += weightVelocities[index];
        }

        // change bias
        biasVelocity = ( momentum * biasVelocity ) - ( realBlame * learningRate );
        bias += biasVelocity;

        return pastBlames;
    }
}