using csnets.Neural.Activations;
namespace csnets.Neural;

public class DenseLayer {

    public List <Neuron> neurons;

    /// <summary>
    /// initializes random neurons for this dense layer with a given Random
    /// </summary>
    /// <param name="numberOfNeurons">the number of neurons to initialize</param>
    /// <param name="numberOfInputs">the number of inputs to each neuron / the number of neurons in the DenseLayer before this one</param>
    /// <param name="random">the random to use</param>
    public DenseLayer ( int numberOfNeurons, int numberOfInputs, Random random ) {
        neurons = [];
        for (int i = 0; i < numberOfNeurons; i++)
        {
            neurons.Add (new Neuron (random, numberOfInputs));
        }
    }

    public DenseLayer ( int numberOfNeurons, int numberOfInputs ) : this ( numberOfNeurons, numberOfInputs, new Random () ) { }


    public List<float> ForwardPass <A> ( List <float> inputs, bool isLastLayer ) where A : IActivation {
        List<float> outputs = [];
        foreach (var neuron in neurons)
        {
            outputs.Add (neuron.ForwardPass <A> ( inputs, !isLastLayer ));
        }
        return outputs;
    }

    public List <float> BackProp <A> ( List <float> inputs, List <float> blame, float learningRate, bool isLastLayer = false ) where A : IActivation {
        List <float> lastLayerBlame = new List <float> (new float[inputs.Count]);

        for (var index = 0; index < neurons.Count; index++)
        {
            var neuron = neurons[index];
            var blameForMe = blame[index];

            float myFinalBlame;
            if (isLastLayer) {
                myFinalBlame = blameForMe;
            } else {
                var rawOutput = neuron.ForwardPass <A> ( inputs, false );
                myFinalBlame = A.Df ( rawOutput ) * blameForMe;
            }

            for (var inputIndex = 0; inputIndex < inputs.Count; inputIndex++)
            {
                var inputBlame = myFinalBlame * neurons[index].weights[inputIndex];
                lastLayerBlame[inputIndex] += inputBlame;
            }

            for (var inputIndex = 0; inputIndex < inputs.Count; inputIndex++)
            {
                var deltaWeight = myFinalBlame * inputs[inputIndex];
                neurons[index].weights[inputIndex] -= deltaWeight * learningRate;
            }

            neurons[index].bias -= myFinalBlame * learningRate;
        }

        return lastLayerBlame;
    }


}