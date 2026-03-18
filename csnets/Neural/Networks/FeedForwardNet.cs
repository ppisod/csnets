using csnets.Neural.Activations;
using csnets.Neural.Initializations;
using csnets.Neural.Loss;
using csnets.Neural.Optimizers;

namespace csnets.Neural.Networks;

public class FeedForwardNet {
    public readonly int inputSize;
    public List <DenseLayer> layers;

    public FeedForwardNet ( int inputs, int numLayers, int outputs, IOptimizer optimizer, int? numNeuronsPerLayer_Min, int? numNeuronsPerLayer_Max, IInitialization? init = null ) {
        inputSize = inputs;
        var random = new Random ();
        var currentInputSize = inputSize;
        numNeuronsPerLayer_Min ??= 5;
        numNeuronsPerLayer_Max ??= 10;
        layers = [];
        for (int i = 0; i < numLayers; i++)
        {

            if (i == numLayers - 1)
            {
                layers.Add (
                    new DenseLayer (
                        outputs,
                        currentInputSize,
                        random,
                        optimizer,
                        init
                    )
                );
                break;
            }

            int k = (int) random.NextInt64 ( numNeuronsPerLayer_Min.Value, numNeuronsPerLayer_Max.Value );
            layers.Add (
                new DenseLayer (
                    k,
                    currentInputSize,
                    random,
                    optimizer,
                    init
                )
            );

            currentInputSize = k;
        }
    }

    public FeedForwardNet (
        int inputs,
        IOptimizer optimizer,
        int[] layers,
        int output,
        IInitialization? init = null
    ) {
        inputSize = inputs;
        var random = new Random ();
        var currentInputSize = inputSize;
        this.layers = [];
        foreach (var layer in layers)
        {
            this.layers.Add ( new DenseLayer (layer, currentInputSize, random, optimizer, init) );
            currentInputSize = layer;
        }
        this.layers.Add ( new DenseLayer (output, currentInputSize, random, optimizer, init) );
    }

    public FeedForwardNet ( int inputSize, List <DenseLayer> layers )  {
        if (layers.Count == 0) throw new Exception ( "Must have at least one layer" );
        // TODO: Check if this is the best way to do it
        foreach (var neuron in layers[0].neurons)
        {
            if (neuron.weights.Length != inputSize) throw new Exception ( "First layer neurons must have same number of inputs as input size." );
        }
        this.inputSize = inputSize;
        this.layers = layers;
    }

    public float[] Run <A> ( float[] inputs ) where A : IActivation {
        if (inputs.Length != inputSize)
        {
            throw new Exception ( "Inputs does not match set input size." );
        }

        float[] lastDenseLayerOutput = inputs;
        for (var index = 0; index < layers.Count; index++)
        {
            var layer = layers[index];
            bool isLast = index == layers.Count - 1;
            lastDenseLayerOutput = layers[index].ForwardPass<A>(lastDenseLayerOutput, isLast);
        }
        return lastDenseLayerOutput;
    }

    public void DebugPrint <A, L> ( float[] inputs, float[] targets ) where A : IActivation where L : ILoss {
        var outputs = Run <A> ( inputs );
        // for (int i = 0; i < outputs.Length; i++)
        // {
        //     var output = outputs[i];
        //     var target = targets[i];
        //     Console.WriteLine ( $"OutputN: {i}, Output: {output}, Target: {target}" );
        // }

        float loss = 0;
        for (int i = 0; i < outputs.Length; i++)
        {
            loss += L.Calculate ( outputs[i], targets[i] );
        }
        loss *= 0.5f;
        Console.WriteLine ( $"Loss: {loss}" );
    }

    public void Train <A, L> ( float[] inputs, float[] targets, float learningRate, bool log = true, bool batching = false ) where A : IActivation where L : ILoss {
        if (inputs.Length != inputSize)
        {
            throw new Exception ( "Inputs does not match set input size." );
        }

        // caching
        List <float[]> layerInputs = new List <float[]> ();
        float[] currentLayerInputs = inputs;

        // Forward Pass
        for (var index = 0; index < layers.Count; index++)
        {
            layerInputs.Add ( currentLayerInputs );
            var layer = layers[index];
            bool isLast = index == layers.Count - 1;
            currentLayerInputs = layer.ForwardPass <A> ( currentLayerInputs, isLast );
        }

        float[] outputs = currentLayerInputs;
        if (outputs.Length != targets.Length)
        {
            throw new Exception ( "Outputs and targets must have same length." );
        }

        // Calculate initial blame from Loss
        float[] blame = new float[outputs.Length];
        for (var index = 0; index < outputs.Length; index++)
        {
            blame[index] = ( L.CalculateDerivative ( outputs[index], targets[index] ) );
        }

        // Backward Pass
        for (var index = layers.Count - 1; index >= 0; index--)
        {
            var layer = layers[index];
            var inputsForThisLayer = layerInputs[index];
            bool isLast = index == layers.Count - 1;
            blame = layer.BackProp <A> ( inputsForThisLayer, blame, learningRate, isLast, batching );
        }

        if (log)
        {
            DebugPrint <A, L> ( inputs, targets );
        }
    }

    public void Train <A, L> (
        float[][] inputs,
        float[][] targets,
        float learningRate,
        bool log = false
    ) where A : IActivation where L : ILoss {
        if (inputs.Length != targets.Length)
        {
            throw new Exception ( "Inputs and targets must have same length." );
        }

        for (int i = 0; i < inputs.Length; i++)
        {

            Train <A, L> ( inputs[i], targets[i], learningRate, log, true );

        }

        foreach (var layer in layers)
        {
            layer.ApplyGradients ( learningRate );
        }
    }

}
