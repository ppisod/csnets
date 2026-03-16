using csnets.Neural;
namespace csnets.Neural.Initializations;

public interface IInitialization {
    Weight[] InitWeights ( Random random, int inputSize );
}
